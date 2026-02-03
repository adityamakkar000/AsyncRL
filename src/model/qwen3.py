from typing import Optional

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from .config import KVCache, QwenConfig
from .utils import convert_dtype, make_attention_mask, make_prompt_mask


class RoPEMatrixCache(nn.Module):
    sequence_len: int
    model_dim: int
    rope_base: int

    def setup(self):
        pos = jnp.arange(0, self.model_dim, 2, dtype=jnp.float32) / self.model_dim
        theta = 1.0 / (self.rope_base**pos)
        inp = jnp.einsum("t,k->tk", jnp.arange(self.sequence_len), theta, precision=jax.lax.Precision.HIGHEST)

        self.sin = jnp.sin(inp).astype(jnp.float32)
        self.cos = jnp.cos(inp).astype(jnp.float32)

    def get_rope_matrix(self, index_map) -> tuple[Array, Array]:
        B, T = index_map.shape
        index_map = index_map.reshape(B * T)

        @jax.vmap
        def get_single_sin_cos_row(input):
            return jax.tree.map(
                lambda x: jax.lax.dynamic_slice_in_dim(x, input, 1, axis=0),
                (self.sin, self.cos),
            )

        return jax.tree.map(lambda x: x.reshape(B, T, 1, self.model_dim // 2), get_single_sin_cos_row(index_map))


def apply_rope(x: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray) -> jnp.ndarray:
    *_, C = x.shape
    x1, x2 = x[..., : C // 2], x[..., C // 2 :]
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    return out


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


class GroupedQueryAttention(nn.Module):
    model_dim: int
    n_heads: int
    n_groups: int
    head_dim: int
    rope_base: int
    activation_dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.n_heads % self.n_groups == 0, "Number of heads must be divisible by number of kv groups"
        assert self.model_dim % self.n_heads == 0, "Model dim must be divisible by number of heads"

        self.kv_group_size = self.n_heads // self.n_groups
        self.d_out = self.n_heads * self.head_dim

    @nn.compact
    def __call__(
        self,
        x: Array,
        sequence_lens: jax.Array,
        mask: Array,
        rope_matrix: tuple[Array, Array],
        kv_cache: Optional[KVCache] = None,
    ):
        B, T, C = x.shape
        t_start = kv_cache.length if kv_cache else 0

        q = nn.Dense(features=self.d_out, use_bias=False, dtype=self.activation_dtype)(x)
        k = nn.Dense(features=self.n_groups * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
        v = nn.Dense(features=self.n_groups * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)

        q = einops.rearrange(q, "... t (h d) -> ... t h d", d=self.head_dim)
        k = einops.rearrange(k, "... t (g d) -> ... t g d", d=self.head_dim)
        v = einops.rearrange(v, "... t (g d) -> ... t g d", d=self.head_dim)

        q = RMSNorm(self.activation_dtype)(q)
        k = RMSNorm(self.activation_dtype)(k)

        q = apply_rope(q, rope_matrix[0], rope_matrix[1])
        k = apply_rope(k, rope_matrix[0], rope_matrix[1])

        if kv_cache:
            k, v = jax.tree.map(
                lambda cache, val: jax.lax.dynamic_update_slice_in_dim(
                    cache, val.astype(cache.dtype), t_start, axis=1
                ).astype(self.activation_dtype),
                (kv_cache.k, kv_cache.v),
                (k, v),
            )
            kv_cache = KVCache(k=k, v=v, length=t_start + T)

        q = einops.rearrange(q, pattern="b t (g r) d -> b t g r d", g=k.shape[-2])

        wei = jnp.einsum("btgrd, bTgd -> btTgr", q.astype(jnp.float32), k.astype(jnp.float32)) * (self.head_dim**-0.5)

        wei = einops.rearrange(wei, pattern="b t T g r -> b (g r) t T ")

        wei = jnp.where(mask == 1, wei, -jnp.inf)
        wei = jax.nn.softmax(wei, axis=-1)
        nan_mask = ~jnp.isnan(wei)
        wei = jnp.where(nan_mask == 1, wei, 0)

        wei = einops.rearrange(tensor=wei, pattern="b (g r ) t T -> b t T g r", g=k.shape[2])

        out = jnp.einsum("btTgr, bTgd -> btgrd", wei, v.astype(jnp.float32))
        out = out.astype(x.dtype)

        out = einops.rearrange(out, "b t g r d -> b t (g r d)").astype(self.activation_dtype)
        out = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(out)

        return out, kv_cache


class Block(nn.Module):
    d_ff: int
    model_dim: int
    n_heads: int
    n_groups: int
    head_dim: int
    rope_base: int
    activation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Array,
        sequence_lens: jax.Array,
        mask: Array,
        rope_matrix: tuple[Array, Array],
        layer_cache: Optional[KVCache] = None,
    ):
        connection_1 = x
        x = RMSNorm(activation_dtype=self.activation_dtype)(x)
        x, out_layer_cache = GroupedQueryAttention(
            model_dim=self.model_dim,
            n_heads=self.n_heads,
            n_groups=self.n_groups,
            head_dim=self.head_dim,
            rope_base=self.rope_base,
            activation_dtype=self.activation_dtype,
        )(x, sequence_lens, mask, rope_matrix, layer_cache)

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
        B, T = x.shape
        embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dim,
            dtype=self.activation_dtype,
            name="token_emb",
        )

        x = embed_layer(x)

        out_cache = []

        rope_cache = RoPEMatrixCache(sequence_len=self.sequence_len, model_dim=self.head_dim, rope_base=self.rope_base)

        t_start = kv_cache[0].length if kv_cache else 0
        prompt_mask = make_prompt_mask(self.sequence_len, cache_len=t_start + T, seq_lens=sequence_lens)
        index_map = jnp.cumsum(prompt_mask, axis=-1)
        # account for kv_cache length
        index_map_with_offset = jnp.where(index_map > 0, index_map + t_start - 1, 0)
        index_map_with_offset_sliced = jax.lax.dynamic_slice(index_map_with_offset, (0, t_start), (B, T))

        sin, cos = rope_cache.get_rope_matrix(index_map_with_offset_sliced)

        attention_mask = make_attention_mask(
            query_shape=T,
            key_shape=T if not kv_cache else self.sequence_len,
            t_start=t_start,
            seq_lens=sequence_lens,
        )

        for i in range(self.n_layers):
            in_layer_cache = kv_cache[i] if kv_cache else None
            x, out_layer_cache = Block(
                d_ff=self.d_ff,
                model_dim=self.model_dim,
                n_heads=self.n_heads,
                n_groups=self.n_groups,
                head_dim=self.head_dim,
                rope_base=self.rope_base,
                activation_dtype=self.activation_dtype,
            )(x, sequence_lens, attention_mask, (sin, cos), in_layer_cache)
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
