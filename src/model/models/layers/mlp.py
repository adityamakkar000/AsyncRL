import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from ...utils import act_constrain


class FeedForward(nn.Module):
    model_dim: int
    d_ff: int
    activation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, shard: bool = False):
        gate = nn.silu(nn.Dense(features=self.d_ff, use_bias=False, dtype=self.activation_dtype)(x))
        gate = act_constrain(gate, shard)

        up = nn.Dense(features=self.d_ff, use_bias=False, dtype=self.activation_dtype)(x)
        up = act_constrain(up, shard)

        h = act_constrain(gate * up, shard)
        return nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(h)
