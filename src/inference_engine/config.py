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

    _max_prompts_decode: int = 16
    _max_decode_batch_size: int = 64

    intial_sequence_len: int = 64
    precompile: bool = True
    kv_cache_dtype: str = "bfloat16"
    reasoning_budget: Optional[int] = None
    think_mode: bool = True
    max_prefill_sequence_len: int = 1024
    system_prompt: bool = False


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

    def get_index(self, i: int) -> "InferenceState":
        return InferenceState(
            next_token=self.next_token[i : i + 1],
            kv_cache=[
                KVCache(k=cache.k[i : i + 1], v=cache.v[i : i + 1], length=cache.length) for cache in self.kv_cache
            ],
            key=self.key[i : i + 1],
            seq_lens=self.seq_lens[i : i + 1],
            stop_mask=self.stop_mask[i : i + 1],
            end_of_think=self.end_of_think[i : i + 1],
            out_tokens=self.out_tokens[i : i + 1],
            out_logprobs=self.out_logprobs[i : i + 1],
        )


@dataclass
class InferenceShardings:
    mesh: jax.sharding.Mesh
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
