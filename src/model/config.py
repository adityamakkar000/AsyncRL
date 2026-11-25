import abc
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from jaxtyping import Array, PyTree
from omegaconf import MISSING
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .utils import dequantize, get_torch_weights_to_jax, quantize, save_to_hf


class Int8Array(struct.PyTreeNode):
    value: jax.Array
    scale: jax.Array

    @classmethod
    def quantize(cls, x: jax.Array) -> "Int8Array":
        return cls(*quantize(x))

    def dequantize(self, dtype: jnp.dtype = jnp.bfloat16) -> jax.Array:
        return dequantize(self.value, self.scale, dtype)

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.dtype(jnp.int8)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def shape_and_dtype(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), (self))


class KVCache(struct.PyTreeNode):
    k: Array | Int8Array  # [l,b,h,t,d]
    v: Array | Int8Array
    length: Array

    @property
    def max_len(self) -> int:
        return self.k.shape[-2]

    @property
    def is_quantized(self) -> bool:
        return isinstance(self.k, Int8Array)

    @property
    def dtype(self) -> jnp.dtype:
        return self.k.dtype

    @property
    def shape_dtype(self):
        if isinstance(self.k, Int8Array):
            return self.k.shape_and_dtype
        return jax.ShapeDtypeStruct(self.k.shape, self.dtype)

    def prefill(self, layer: Array, k: Array, v: Array) -> "KVCache":
        def _update_cache(cache: Array | Int8Array, new: Array) -> Array | Int8Array:
            update = Int8Array.quantize(new) if isinstance(cache, Int8Array) else new
            return jax.tree.map(
                lambda c, n: jax.lax.dynamic_update_slice(c, n[None].astype(c.dtype), (layer,) + (0,) * (c.ndim - 1)),
                cache,
                update,
            )

        return self.replace(k=_update_cache(self.k, k), v=_update_cache(self.v, v))


class AttentionMask(NamedTuple):
    q_segment_ids: Array
    k_segment_ids: Array


class RopeMatrix(NamedTuple):
    sin: Array
    cos: Array


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

    @staticmethod
    def eos_token_id(tokenizer: PreTrainedTokenizerBase) -> int | None:
        """Token that ends a generation."""
        return cast(int | None, tokenizer.eos_token_id)

    @staticmethod
    def get_chat_template(system_prompt: str | None, prompt: str) -> list[dict[str, str]]:
        system = [{"role": "system", "content": system_prompt}] if system_prompt else []
        return [*system, {"role": "user", "content": prompt}]

    @classmethod
    def apply_chat_template(
        cls, tokenizer: PreTrainedTokenizerBase, system_prompt: str | None, prompt: str, enable_thinking: bool = True
    ) -> list[int]:
        tokens = tokenizer.apply_chat_template(
            cls.get_chat_template(system_prompt, prompt),
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=True,
            return_dict=False,
        )
        return cast(list[int], tokens)

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
