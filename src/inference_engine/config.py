from dataclasses import dataclass
from typing import Optional

from flax import struct
from jaxtyping import Array, PyTree
from omegaconf import MISSING

from src.model import KVCache


@dataclass
class InferenceConfig:
    temperature: float = MISSING
    top_k: Optional[int] = MISSING
    top_p: Optional[float] = MISSING
    max_seq_len: int = MISSING
    batch_size: int = MISSING
    group_size: int = MISSING
    kv_cache_dtype: str = MISSING
    n_replicas: int = 1
    intial_sequence_len: int = 64
    precompile: bool = True


@struct.dataclass
class InferenceState:
    next_token: Array
    next_probs: Array
    kv_cache: list[KVCache]
    key: Array
    seq_lens: Array
    params: PyTree
    stop_mask: Array


@dataclass
class InferenceRollout:
    rollouts: list[Array]
    logprobs: list[Array]


@dataclass
class InferenceResults:
    rollouts: list[InferenceRollout]
    output_strs: Optional[list[list[str]] | list[str]]
    metrics: PyTree
