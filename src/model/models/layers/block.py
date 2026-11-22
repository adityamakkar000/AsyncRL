import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from ...config import AttentionMask, KVCache, RopeMatrix
from ...utils import act_constrain
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
        carry: tuple[Array, KVCache | None],
        masks: AttentionMask,
        rope_matrix: RopeMatrix,
        layer: Array,
    ):
        x, layer_cache = carry
        shard = layer_cache is None

        x = act_constrain(x, shard)
        h = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(x)
        h, out_layer_cache = GroupedQueryAttention(
            model_dim=self.model_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            activation_dtype=self.activation_dtype,
            qk_norm=self.qk_norm,
            rms_eps=self.rms_eps,
        )(h, masks, rope_matrix, None if layer_cache is None else (layer, layer_cache))
        x = act_constrain(x + h, shard)

        h = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(x)
        x = x + FeedForward(model_dim=self.model_dim, d_ff=self.d_ff, activation_dtype=self.activation_dtype)(h, shard)

        return (act_constrain(x, shard), out_layer_cache), None


RematBlock = nn.remat(Block)
