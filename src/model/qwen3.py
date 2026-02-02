from typing import Optional

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from .config import KVCache, QwenConfig
from .utils import convert_dtype, make_attention_mask, make_prompt_mask


class FeedForward(nn.Module):
    d_ff: int
    model_dim: int
    activation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array):
        x_fc1 = nn.Dense(features=self.d_ff, use_bias=False, dtype=self.activation_dtype)(x)
        x_fc2 = nn.Dense(features=self.d_ff, use_bias=False, dtype=self.activation_dtype)(x)
        x = nn.silu(x_fc1) * x_fc2
        x_fc3 = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(x)

        return x_fc3


class RMSNorm(nn.Module):
    activation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array):
        rms = jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6)
        gamma = self.param("gamma", nn.initializers.ones, (x.shape[-1]), self.activation_dtype)
        x = (x * gamma) / rms
        return x


class RoPE(nn.Module):
    sequence_len: int
    model_dim: int
    rope_base: int

    def setup(self):
        assert self.model_dim % 2 == 0, "Hidden dimension must be even"

        m = jnp.arange(0, self.sequence_len, dtype=jnp.float32)
        pos = jnp.arange(0, self.model_dim, 2, dtype=jnp.float32) / self.model_dim
        theta = 1.0 / (self.rope_base**pos)

        inp = jnp.einsum("t,k->tk", m, theta, precision=jax.lax.Precision.HIGHEST)

        self.sin = jnp.sin(inp).astype(jnp.float32)
        self.cos = jnp.cos(inp).astype(jnp.float32)

    def __call__(self, x: Array, index_map: Array):
        x = einops.rearrange(x, "b t g d -> b g t d")
        B, h, T, C = x.shape

        index_map = index_map.reshape(-1)

        @jax.vmap
        def get_single_sin_cos_row(input):
            sin = jax.lax.dynamic_slice(self.sin, (input, 0), (1, C // 2))
            cos = jax.lax.dynamic_slice(self.cos, (input, 0), (1, C // 2))
            return sin, cos

        sin, cos = jax.tree.map(lambda x: x.reshape(B, T, C // 2)[:, None, ...], get_single_sin_cos_row(index_map))

        x1, x2 = x[..., : C // 2], x[..., C // 2 :]

        out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

        out = einops.rearrange(out, pattern="b g t d -> b t g d")
        return out


class GroupedQueryAttention(nn.Module):
    model_dim: int
    n_heads: int
    n_groups: int
    max_sequence_len: int
    head_dim: int
    rope_base: int
    activation_dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.n_heads % self.n_groups == 0, "Number of heads must be divisible by number of kv groups"
        assert self.model_dim % self.n_heads == 0, "Model dim must be divisible by number of heads"

        self.kv_group_size = self.n_heads // self.n_groups
        self.d_out = self.n_heads * self.head_dim

    @nn.compact
    def __call__(self, x: Array, seq_lens: jax.Array, kv_cache: Optional[KVCache] = None):
        B, T, C = x.shape
        t_start = kv_cache.length if kv_cache else 0

        q = nn.Dense(features=self.d_out, use_bias=False, dtype=jnp.float32)(x)
        k = nn.Dense(features=self.n_groups * self.head_dim, use_bias=False, dtype=jnp.float32)(x)
        v = nn.Dense(features=self.n_groups * self.head_dim, use_bias=False, dtype=jnp.float32)(x)

        q = einops.rearrange(q, "... t (h d) -> ... t h d", d=self.head_dim)
        k = einops.rearrange(k, "... t (g d) -> ... t g d", d=self.head_dim)
        v = einops.rearrange(v, "... t (g d) -> ... t g d", d=self.head_dim)

        q = RMSNorm(self.activation_dtype)(q)
        k = RMSNorm(self.activation_dtype)(k)

        prompt_mask = make_prompt_mask(T, seq_lens)
        index_map = jnp.cumsum(prompt_mask, axis=-1)
        # account for kv_cache length
        index_map_with_offset = jnp.where(index_map > 0, index_map + t_start - 1, 0)

        q = RoPE(self.max_sequence_len, self.head_dim, self.rope_base)(q, index_map_with_offset)
        k = RoPE(self.max_sequence_len, self.head_dim, self.rope_base)(k, index_map_with_offset)

        if kv_cache:
            k_cache, v_cache = kv_cache.k, kv_cache.v

            k, v = jax.tree.map(
                lambda cache, val: jax.lax.dynamic_update_slice_in_dim(cache, val.astype(cache.dtype), t_start, axis=1),
                (k_cache, v_cache),
                (k, v),
            )
            kv_cache = KVCache(k=k_cache, v=v_cache, length=t_start + T)

        q = einops.rearrange(q, pattern="b t (g r) d -> b t g r d", g=k.shape[-2])

        wei = jnp.einsum("btgrd, bTgd -> btTgr", q, k) * (self.head_dim**-0.5)

        wei = einops.rearrange(wei, pattern="b t T g r -> b (g r) t T ")

        attention_mask = make_attention_mask(t=T, T=k.shape[1], seq_lens=seq_lens)
        wei = jnp.where(attention_mask == 1, wei, -jnp.inf)
        wei = jax.nn.softmax(wei, axis=-1)
        nan_mask = ~jnp.isnan(wei)
        wei = jnp.where(nan_mask == 1, wei, 0)

        wei = einops.rearrange(tensor=wei, pattern="b (g r ) t T -> b t T g r", g=k.shape[2])

        out = jnp.einsum("btTgr, bTgd -> btgrd", wei, v)
        out = out.astype(x.dtype)

        out = einops.rearrange(out, "b t g r d -> b t (g r d)")
        out = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(out)

        return out, kv_cache


class Block(nn.Module):
    d_ff: int
    sequence_len: int
    model_dim: int
    n_heads: int
    n_groups: int
    head_dim: int
    rope_base: int
    activation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, seq_lens: jax.Array, layer_cache: Optional[KVCache] = None):
        connection_1 = x
        x = RMSNorm(activation_dtype=self.activation_dtype)(x)
        x, out_layer_cache = GroupedQueryAttention(
            model_dim=self.model_dim,
            n_heads=self.n_heads,
            n_groups=self.n_groups,
            max_sequence_len=self.sequence_len,
            head_dim=self.head_dim,
            rope_base=self.rope_base,
            activation_dtype=self.activation_dtype,
        )(x, seq_lens, layer_cache)

        x = x + connection_1

        connection_2 = x
        x = RMSNorm(activation_dtype=self.activation_dtype)(x)
        x = FeedForward(d_ff=self.d_ff, model_dim=self.model_dim, activation_dtype=self.activation_dtype)(x)
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
    rope_base: int
    activation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, sequence_lens: jax.Array, kv_cache: Optional[list[KVCache]] = None):
        embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dim,
            dtype=self.activation_dtype,
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
                rope_base=self.rope_base,
                activation_dtype=self.activation_dtype,
            )(x, sequence_lens, in_layer_cache)

            out_cache.append(out_layer_cache)

        x = RMSNorm(activation_dtype=self.activation_dtype)(x)

        # logits = embed_layer.attend(x)
        logits = nn.Dense(features=self.vocab_size, use_bias=False, dtype=self.activation_dtype)(x)
        return logits, out_cache if kv_cache else None

    @classmethod
    def from_config(cls, config: QwenConfig):
        activation_dtype = convert_dtype(config.activation_dtype)
        return cls(
            vocab_size=config.vocab_size,
            d_ff=config.d_ff,
            sequence_len=config.sequence_len,
            model_dim=config.model_dim,
            n_heads=config.n_heads,
            n_groups=config.n_groups,
            head_dim=config.head_dim,
            n_layers=config.n_layers,
            rope_base=config.rope_base,
            activation_dtype=activation_dtype,
        )
