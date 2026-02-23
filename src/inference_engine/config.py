from dataclasses import dataclass
from typing import Optional

import jax
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
    n_replicas: int = 1
    intial_sequence_len: int = 64
    precompile: bool = True
    kv_cache_dtype: str = "bfloat16"
    reasoning_budget: Optional[int] = None
    max_prefill_sequence_len: int = 1024


@struct.dataclass
class InferenceState:
    next_token: Array
    kv_cache: list[KVCache]
    key: Array
    seq_lens: Array
    stop_mask: Array
    end_of_think: Array
    out_tokens: Array
    out_logprobs: Array


@dataclass
class InferenceShardings:
    split_sharding: jax.NamedSharding
    replicate_sharding: jax.NamedSharding
    kv_cache_sharding: KVCache
    state_sharding: InferenceState
    params_sharding: PyTree
    prefill_shardings: dict[str, PyTree]
    decode_shardings: dict[str, PyTree]


@dataclass
class InferenceRollout:
    rollouts: list[Array]
    logprobs: list[Array]


@dataclass
class InferenceResults:
    rollouts: list[InferenceRollout]
    output_strs: list[list[str]]
    metrics: PyTree
