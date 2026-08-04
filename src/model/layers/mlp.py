import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array


class FeedForward(nn.Module):
    model_dim: int
    d_ff: int
    activation_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array):
        gate = nn.silu(nn.Dense(features=self.d_ff, use_bias=False, dtype=self.activation_dtype)(x))
        up = nn.Dense(features=self.d_ff, use_bias=False, dtype=self.activation_dtype)(x)
        return nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(gate * up)
