import math

import jax
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def fit_block(block: int, dim: int) -> int:
    while block > NUM_LANES and (block > dim or dim % block != 0):
        block //= 2
    return block


def check_div(n: int, bs: int):
    if n % bs != 0:
        raise ValueError(f"size of {n} could not be divided by {bs}")


def verify_block(block_name, dim_name, block, dim, should_divide=True):
    if block > dim:
        raise ValueError(f"{block_name}={block} should be smaller or equal to {dim_name}={dim}")
    if should_divide and dim % block != 0:
        raise ValueError(f"{dim_name}={dim} should be divisible by {block_name}={block}")


def nbytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
    return math.prod(x.shape) * x.dtype.itemsize
