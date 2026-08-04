from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from .config import BaseModel, KVCache
from .layers import RematBlock, RMSNorm, RopeCorrection, gather_rope, no_correction, rope_tables
from .utils import make_attention_mask, make_prompt_mask


class Transformer(BaseModel):
    vocab_size: int
    sequence_len: int
    model_dim: int
    d_ff: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_layers: int
    rope_base: int
    activation_dtype: jnp.dtype = jnp.float32
    tie_weights: bool = False
    qk_norm: bool = False
    rms_eps: float = 1e-6
    rope_correction: RopeCorrection = no_correction

    @nn.compact
    def __call__(
        self,
        x: Array,
        sequence_lens: jax.Array,
        kv_cache: Optional[list[KVCache]] = None,
    ) -> tuple[Array, list[KVCache]]:
        B, T = x.shape
        embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dim,
            dtype=jnp.float32,
            name="token_emb",
        )
        x = embed_layer(x)

        t_start = kv_cache[0].length if kv_cache else 0
        attention_len = kv_cache[0].k.shape[1] if kv_cache else T

        prompt_mask = make_prompt_mask(attention_len, cache_len=t_start + T, seq_lens=sequence_lens)
        index_map = jnp.cumsum(prompt_mask, axis=-1)
        index_map = jnp.where(index_map > 0, index_map - 1, 0)
        index_map = jax.lax.dynamic_slice(index_map, (0, t_start), (B, T))

        sin_table, cos_table = rope_tables(self.sequence_len, self.head_dim, self.rope_base, self.rope_correction)
        rope_matrix = gather_rope(sin_table, cos_table, index_map)

        attention_mask = (
            prompt_mask
            if kv_cache is None
            else make_attention_mask(
                query_shape=T,
                key_shape=attention_len,
                t_start=t_start,
                seq_lens=sequence_lens,
            )
        )

        out_cache: list[KVCache] = []
        for i in range(self.n_layers):
            x, out_layer_cache = RematBlock(
                model_dim=self.model_dim,
                d_ff=self.d_ff,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                activation_dtype=self.activation_dtype,
                qk_norm=self.qk_norm,
                rms_eps=self.rms_eps,
                name=f"Block_{i}",
            )(x, attention_mask, rope_matrix, kv_cache[i] if kv_cache else None)
            out_cache.append(out_layer_cache)

        x = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(x)

        if self.tie_weights:
            logits = embed_layer.attend(x)
        else:
            logits = nn.Dense(features=self.vocab_size, use_bias=False, dtype=jnp.float32)(x)

        return logits.astype(jnp.float32), out_cache

    @property
    def seq_len(self) -> int:
        return self.sequence_len

    @property
    def kv_shape(self):
        return (self.n_layers, self.n_kv_heads, self.head_dim)
