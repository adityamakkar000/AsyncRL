from dataclasses import dataclass

from flax import struct
from jaxtyping import Array
from omegaconf import MISSING


@struct.dataclass
class KVCache:
    k: Array
    v: Array
    length: int


@dataclass
class modelConfig:
    vocab_size: int = MISSING
    d_ff: int = MISSING
    sequence_len: int = MISSING
    model_dim: int = MISSING
    n_heads: int = MISSING
    n_groups: int = MISSING
    head_dim: int = MISSING
    n_layers: int = MISSING
    model_dtype: str = "float32"
