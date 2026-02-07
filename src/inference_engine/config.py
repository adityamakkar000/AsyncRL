from dataclasses import dataclass

from flax import struct
from jaxtyping import Array, PyTree
from omegaconf import MISSING

from src.model import KVCache


@dataclass
class InferenceConfig:
    temperature: float = MISSING
    top_k: int = MISSING
    top_p: float = MISSING
    max_seq_len: int = MISSING
    batch_size: int = MISSING
    group_size: int = MISSING
    kv_cache_dtype: str = MISSING


@struct.dataclass
class InferenceState:
    next_token: Array
    kv_cache: list[KVCache]
    key: Array
    seq_lens: Array
    params: PyTree


@dataclass
class InferenceRollout:
    prompt: str
    rollouts: list[str]
    logprobs: list[Array]
