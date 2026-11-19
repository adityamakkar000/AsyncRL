import abc
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
from flax import linen as nn
from flax import struct
from jaxtyping import Array, PyTree
from omegaconf import MISSING

from .utils import get_torch_weights_to_jax, save_to_hf


class AttentionMask(NamedTuple):
    q_segment_ids: Array
    k_segment_ids: Array


class RopeMatrix(NamedTuple):
    sin: Array
    cos: Array


class KVCache(struct.PyTreeNode):
    k: Array
    v: Array
    length: Array

    @property
    def max_len(self) -> int:
        return self.k.shape[-2]


@dataclass
class ModelConfig:
    hf_model_name: str
    model_args: dict[str, Any] = MISSING
    fused_chunk_size: int | None = 1024


class BaseModel(abc.ABC, nn.Module):
    @property
    @abc.abstractmethod
    def seq_len(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def activation_dtype(self):
        pass

    @property
    @abc.abstractmethod
    def kv_shape(self):
        pass

    @property
    @abc.abstractmethod
    def hf_mapping(self):
        pass

    @property
    @abc.abstractmethod
    def reverse_hf_mapping(self):
        pass

    def load_from_hf(self, params: PyTree, model_name: str) -> PyTree:
        return get_torch_weights_to_jax(params, model_name, self.hf_mapping)

    def save_to_hf(self, path: str, params: PyTree, model_name: str) -> None:
        params = dict(params)
        stacked = params.pop("Block")
        n_layers = jax.tree.leaves(stacked)[0].shape[0]
        for i in range(n_layers):
            params[f"Block_{i}"] = jax.tree.map(lambda p, i=i: p[i], stacked)

        save_to_hf(path, params, model_name, self.reverse_hf_mapping)

    @abc.abstractmethod
    def __call__(self, x: Array, sequence_lens: Array, kv_cache: KVCache | None = None) -> tuple[Array, KVCache | None]:
        pass
