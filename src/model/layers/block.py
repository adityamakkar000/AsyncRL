from typing import Optional

import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from ..config import KVCache
from .attention import GroupedQueryAttention
from .mlp import FeedForward
from .norm import RMSNorm


class Block(nn.Module):
    model_dim: int
    d_ff: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    activation_dtype: jnp.dtype = jnp.float32
    qk_norm: bool = False
    rms_eps: float = 1e-6

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Array,
        rope_matrix: tuple[Array, Array],
        layer_cache: Optional[KVCache] = None,
    ):
        h = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(x)
        h, out_layer_cache = GroupedQueryAttention(
            model_dim=self.model_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            activation_dtype=self.activation_dtype,
            qk_norm=self.qk_norm,
            rms_eps=self.rms_eps,
        )(h, mask, rope_matrix, layer_cache)
        x = x + h

        h = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(x)
        x = x + FeedForward(model_dim=self.model_dim, d_ff=self.d_ff, activation_dtype=self.activation_dtype)(h)

        return x, out_layer_cache


RematBlock = nn.remat(Block)
