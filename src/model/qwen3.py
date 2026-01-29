from typing import Optional

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from .config import KVCache, QwenConfig
from .utils import convert_dtype, make_attention_mask


class FeedForward(nn.Module):
    d_ff: int
    model_dim: int
    model_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array):
        x_fc1 = nn.Dense(features=self.d_ff, use_bias=False, dtype=self.model_dtype)(x)
        x_fc2 = nn.Dense(features=self.d_ff, use_bias=False, dtype=self.model_dtype)(x)
        x = nn.silu(x_fc1) * x_fc2
        x_fc3 = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.model_dtype)(x)

        return x_fc3


class RMSNorm(nn.Module):
    model_dtype: jnp.dtype = jnp.float32
    shift: bool = False

    @nn.compact
    def __call__(self, x: Array):
        eps = 1e-6
        x /= jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)

        gamma = self.param("gamma", nn.initializers.ones, (x.shape[-1]), self.model_dtype)
        beta = self.param("beta", nn.initializers.ones, (x.shape[-1]), self.model_dtype) if self.shift else None

        x = x * gamma

        if self.shift:
            x += beta

        return x


class RoPE(nn.Module):
    sequence_len: int
    model_dim: int
    model_dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.model_dim % 2 == 0, "Hidden dimension must be even"

        m = jnp.arange(0, self.sequence_len, dtype=self.model_dtype)
        pos = jnp.arange(0, self.model_dim, 2, dtype=self.model_dtype) / self.model_dim
        theta = 1.0 / (1000000**pos)

        inp = jnp.einsum("t,k->tk", m, theta)

        self.sin = jnp.sin(inp)
        self.cos = jnp.cos(inp)

    def __call__(self, x: Array, t_start: int):
        x = einops.rearrange(x, "b t g d -> b g t d")
        B, h, T, C = x.shape

        cos = jax.lax.dynamic_slice(self.cos, (t_start, 0), (T, self.cos.shape[-1]))[None, None]
        sin = jax.lax.dynamic_slice(self.sin, (t_start, 0), (T, self.sin.shape[-1]))[None, None]

        x1, x2 = x[..., : C // 2], x[..., C // 2 :]

        out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

        out = einops.rearrange(out, pattern="b g t d -> b t g d")
        return out


class GroupedQueryAttention(nn.Module):
    model_dim: int
    n_heads: int
    n_groups: int
    q_norm: bool
    k_norm: bool
    max_sequence_len: int
    head_dim: int = 64
    model_dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.n_heads % self.n_groups == 0, "Number of heads must be divisible by number of kv groups"
        assert self.model_dim % self.n_heads == 0, "Model dim must be divisible by number of heads"

        self.kv_group_size = self.n_heads // self.n_groups
        self.d_out = self.n_heads * self.head_dim

    @nn.compact
    def __call__(self, x: Array, seq_lens: jax.Array, kv_cache: Optional[KVCache] = None):
        B, T, C = x.shape
        t_start = kv_cache.length if kv_cache else 0

        q = nn.Dense(features=self.d_out, use_bias=False, dtype=self.model_dtype)(x)
        k = nn.Dense(
            features=self.n_groups * self.head_dim,
            use_bias=False,
            dtype=self.model_dtype,
        )(x)
        v = nn.Dense(
            features=self.n_groups * self.head_dim,
            use_bias=False,
            dtype=self.model_dtype,
        )(x)

        q = einops.rearrange(q, "... t (h d) -> ... t h d", d=self.head_dim)
        k = einops.rearrange(k, "... t (g d) -> ... t g d", d=self.head_dim)
        v = einops.rearrange(v, "... t (g d) -> ... t g d", d=self.head_dim)

        if self.q_norm:
            q = RMSNorm(self.model_dtype)(q)

        if self.k_norm:
            k = RMSNorm(self.model_dtype)(k)

        queries = RoPE(self.max_sequence_len, q.shape[-1], self.model_dtype)(q, t_start)
        k = RoPE(self.max_sequence_len, k.shape[-1], self.model_dtype)(k, t_start)

        if kv_cache:
            k_cache, v_cache = kv_cache.k, kv_cache.v

            k, v = jax.tree.map(
                lambda cache, val: jax.lax.dynamic_update_slice_in_dim(cache, val.astype(cache.dtype), t_start, axis=1),
                (k_cache, v_cache),
                (k, v),
            )
            kv_cache = KVCache(k=k_cache, v=v_cache, length=t_start + T)

        queries = einops.rearrange(queries, pattern="b t (g r) d -> b t g r d", g=k.shape[2])

        wei = jnp.einsum(
            "btgrd, bTgd -> btTgr",
            queries.astype(jnp.float32),
            k.astype(jnp.float32),
        ) / jnp.sqrt(self.head_dim)

        wei = einops.rearrange(wei, pattern="b t T g r -> b t T (g r)")

        attention_mask = make_attention_mask(t=T, T=k.shape[1], seq_lens=seq_lens)
        wei = jnp.where(attention_mask == 1, wei, -jnp.inf)
        wei = jax.nn.softmax(wei, axis=-2)
        nan_mask = ~jnp.isnan(wei)
        wei = jnp.where(nan_mask == 1, wei, 0)

        wei = einops.rearrange(tensor=wei, pattern="b t T (g r) -> b t T g r", g=k.shape[2])

        out = jnp.einsum("btTgr, bTgd -> btgrd", wei, v.astype(jnp.float32))
        out = einops.rearrange(out, "b t g r d -> b t (g r d)")
        out = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.model_dtype)(out)
        return out, kv_cache


class Block(nn.Module):
    d_ff: int
    sequence_len: int
    model_dim: int
    n_heads: int
    n_groups: int
    head_dim: int
    model_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, seq_lens: jax.Array, layer_cache: Optional[KVCache] = None):
        connection_1 = x
        x = RMSNorm(model_dtype=self.model_dtype)(x)
        x, out_layer_cache = GroupedQueryAttention(
            model_dim=self.model_dim,
            n_heads=self.n_heads,
            n_groups=self.n_groups,
            q_norm=True,
            k_norm=True,
            max_sequence_len=self.sequence_len,
            head_dim=self.head_dim,
            model_dtype=self.model_dtype,
        )(x, seq_lens, layer_cache)

        x = x + connection_1

        connection_2 = x
        x = RMSNorm(model_dtype=self.model_dtype)(x)
        x = FeedForward(d_ff=self.d_ff, model_dim=self.model_dim, model_dtype=self.model_dtype)(x)
        x = x + connection_2

        return x, out_layer_cache


class Qwen3(nn.Module):
    vocab_size: int
    d_ff: int
    sequence_len: int
    model_dim: int
    n_heads: int
    n_groups: int
    head_dim: int
    n_layers: int
    model_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, sequence_lens: jax.Array, kv_cache: Optional[list[KVCache]] = None):
        embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dim,
            dtype=self.model_dtype,
            name="token_emb",
        )

        x = embed_layer(x)

        out_cache = []
        for i in range(self.n_layers):
            in_layer_cache = kv_cache[i] if kv_cache else None
            x, out_layer_cache = Block(
                d_ff=self.d_ff,
                sequence_len=self.sequence_len,
                model_dim=self.model_dim,
                n_heads=self.n_heads,
                n_groups=self.n_groups,
                head_dim=self.head_dim,
                model_dtype=self.model_dtype,
            )(x, sequence_lens, in_layer_cache)

            out_cache.append(out_layer_cache)

        x = RMSNorm(model_dtype=self.model_dtype)(x)

        # logits = embed_layer.attend(x)
        logits = nn.Dense(features=self.vocab_size, use_bias=False, dtype=self.model_dtype)(x)
        return logits, out_cache if kv_cache else None

    @classmethod
    def from_config(cls, config: QwenConfig):
        model_dtype = convert_dtype(config.model_dtype)
        return cls(
            vocab_size=config.vocab_size,
            d_ff=config.d_ff,
            sequence_len=config.sequence_len,
            model_dim=config.model_dim,
            n_heads=config.n_heads,
            n_groups=config.n_groups,
            head_dim=config.head_dim,
            n_layers=config.n_layers,
            model_dtype=model_dtype,
        )
