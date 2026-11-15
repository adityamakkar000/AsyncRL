from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

RopeCorrection = Callable[[Array], Array]


def no_correction(freq: Array) -> Array:
    return freq


def rope_tables(
    sequence_len: int,
    head_dim: int,
    rope_base: int,
    correction: RopeCorrection = no_correction,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[Array, Array]:
    pos = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    theta = correction(1.0 / (rope_base**pos))
    inp = jnp.einsum("t,k->tk", jnp.arange(sequence_len), theta, precision=jax.lax.Precision.HIGHEST)
    return jnp.sin(inp).astype(dtype), jnp.cos(inp).astype(dtype)


def gather_rope(sin: Array, cos: Array, index_map: Array) -> tuple[Array, Array]:
    return sin[index_map][:, None, :, :], cos[index_map][:, None, :, :]


def apply_rope(x: Array, sin: Array, cos: Array) -> Array:
    *_, C = x.shape
    x1, x2 = x[..., : C // 2], x[..., C // 2 :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
