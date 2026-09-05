import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from .config import BaseModel, KVCache
from .layers import RematBlock, RMSNorm, RopeCorrection, gather_rope, no_correction, rope_tables
from .utils import dynamic_slice_rows, make_prompt_mask


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
        kv_cache: list[KVCache] | None = None,
        apply_lm_head: bool = True,
    ) -> tuple[Array, list[KVCache]]:
        B, T = x.shape
        embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dim,
            dtype=jnp.float32,
            name="token_emb",
        )
        x = embed_layer(x)

        t_start = kv_cache[0].length if kv_cache else jnp.zeros((B,), dtype=jnp.int32)
        attention_len = kv_cache[0].k.shape[1] if kv_cache else T

        prompt_mask = make_prompt_mask(attention_len, cache_len=t_start + T, seq_lens=sequence_lens)
        masks = (dynamic_slice_rows(prompt_mask, t_start, T), prompt_mask)
        index_map = jnp.cumsum(prompt_mask, axis=-1)
        index_map = jnp.where(index_map > 0, index_map - 1, 0)
        index_map = dynamic_slice_rows(index_map, t_start, T)

        sin_table, cos_table = rope_tables(self.sequence_len, self.head_dim, self.rope_base, self.rope_correction)
        rope_matrix = gather_rope(sin_table, cos_table, index_map)

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
            )(x, masks, rope_matrix, kv_cache[i] if kv_cache else None)
            out_cache.append(out_layer_cache)

        x = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(x)

        if apply_lm_head:
            if self.tie_weights:
                x = embed_layer.attend(x)
            else:
                x = nn.Dense(features=self.vocab_size, use_bias=False, dtype=jnp.float32)(x)

        return x.astype(jnp.float32), out_cache

    @property
    def seq_len(self) -> int:
        return self.sequence_len

    @property
    def kv_shape(self):
        return (self.n_layers, self.n_kv_heads, self.head_dim)
