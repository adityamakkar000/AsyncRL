import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional

from flax import linen as nn
from flax import struct
from jaxtyping import Array, PyTree
from omegaconf import MISSING

from .utils import get_torch_weights_to_jax, save_to_hf


@struct.dataclass
class KVCache:
    k: Array
    v: Array
    length: int


@dataclass
class ModelConfig:
    hf_model_name: str
    model_args: Dict[str, Any] = MISSING


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
        save_to_hf(path, params, model_name, self.reverse_hf_mapping)

    @abc.abstractmethod
    def __call__(
        self, x: Array, sequence_lens: Array, kv_cache: Optional[list[KVCache]] = None
    ) -> tuple[Array, list[KVCache]]:
        pass
