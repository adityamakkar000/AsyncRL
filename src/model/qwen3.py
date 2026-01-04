import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from .config import KVCache
from .model import Block, RMSNorm


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
    def __call__(self, x: Array, seq_lens: jax.Array, kv_cache: list[KVCache]):
        embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dim,
            dtype=self.model_dtype,
            name="token_emb",
        )

        x = embed_layer(x)

        out_cache = []
        for i in range(self.n_layers):
            in_layer_cache = kv_cache[i]
            x, out_layer_cache = Block(
                d_ff=self.d_ff,
                sequence_len=self.sequence_len,
                model_dim=self.model_dim,
                n_heads=self.n_heads,
                n_groups=self.n_groups,
                head_dim=self.head_dim,
                model_dtype=self.model_dtype,
            )(x, seq_lens, in_layer_cache, layer_ind=i)

            out_cache.append(out_layer_cache)

        x = RMSNorm(model_dtype=self.model_dtype)(x)

        # logits = embed_layer.attend(x)
        logits = nn.Dense(features=self.vocab_size, use_bias=False, dtype=self.model_dtype)(x)
        return logits, kv_cache
