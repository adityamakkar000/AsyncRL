import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from ...utils import ACT_SPEC, TP_IN_SPEC, TP_UP_SPEC, act_constrain


class FeedForward(nn.Module):
    model_dim: int
    d_ff: int
    activation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, shard: bool = False):
        x = act_constrain(x, shard, TP_IN_SPEC)

        gate = nn.silu(nn.Dense(features=self.d_ff, use_bias=False, dtype=self.activation_dtype)(x))
        gate = act_constrain(gate, shard, TP_UP_SPEC)

        up = nn.Dense(features=self.d_ff, use_bias=False, dtype=self.activation_dtype)(x)
        up = act_constrain(up, shard, TP_UP_SPEC)

        h = up * gate

        out = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(h)

        out = act_constrain(out, shard, ACT_SPEC)
        return out
