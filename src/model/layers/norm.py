import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array


class RMSNorm(nn.Module):
    activation_dtype: jnp.dtype = jnp.float32
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: Array):
        rms = jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + self.eps)
        gamma = self.param("gamma", nn.initializers.ones, (x.shape[-1],), self.activation_dtype)
        return (x * gamma) / rms
