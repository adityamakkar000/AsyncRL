import glob
import os
import re
import shutil
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch
from huggingface_hub import snapshot_download
from jax.sharding import PartitionSpec as P
from jax.tree_util import DictKey
from jaxtyping import Array, PyTree
from safetensors import safe_open
from safetensors.torch import save_file
from stax.sharding import AXIS_NAMES

from src.constants import HF_CHECKPOINT_PATH

BATCH_AXES = AXIS_NAMES.data_parallel_mesh()
CONTEXT_AXES = AXIS_NAMES.context_parallel_mesh()

ACT_SPEC = P(BATCH_AXES, (*CONTEXT_AXES, AXIS_NAMES.TP), None)

TP_IN_SPEC = P(BATCH_AXES, CONTEXT_AXES, None)
TP_UP_SPEC = P(BATCH_AXES, CONTEXT_AXES, AXIS_NAMES.TP)

QKV_SPEC = P(BATCH_AXES, (AXIS_NAMES.CP_ULYSSES, AXIS_NAMES.TP), AXIS_NAMES.CP_RING, None)
OUT_SPEC = P(BATCH_AXES, None, CONTEXT_AXES, AXIS_NAMES.TP)


def has_mesh() -> bool:
    return not jax.sharding.get_abstract_mesh().empty


def mesh_size() -> int:
    return jax.sharding.get_abstract_mesh().size if has_mesh() else 1


def act_constrain(x: PyTree[Array], shard: bool = True, spec: P = ACT_SPEC) -> PyTree[Array]:
    return jax.lax.with_sharding_constraint(x, spec) if shard and has_mesh() else x


def quantize(x: Array) -> tuple[Array, Array]:
    x = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale = (amax / 127.0 + jnp.finfo(x.dtype).tiny).astype(jnp.bfloat16)
    q = jnp.clip(jnp.round(x / scale), -127, 127).astype(jnp.int8)
    return q, scale[..., 0]  # [x.shape, x.shape[:-1]]


def dequantize(q: Array, scale: Array, dtype: jnp.dtype = jnp.bfloat16) -> Array:
    return q.astype(dtype) * scale[..., None].astype(dtype)


def convert_dtype(dtype_str: str) -> jnp.dtype:
    match dtype_str:
        case "float32":
            return jnp.float32
        case "bfloat16":
            return jnp.bfloat16
        case "float16":
            return jnp.float16
        case "int8":
            return jnp.int8
        case _:
            raise ValueError(f"Unsupported dtype string: {dtype_str}")


def download_hf_weights(name: str, path: str):
    snapshot_download(
        repo_id=name,
        local_dir=path,
        token=os.environ.get("HF_TOKEN", None),
        ignore_patterns=["original/*"],
    )


def get_jax_key(main_key: str, hf_mapping) -> str | None:
    """Convert Hugging Face parameter key to JAX parameter key using the mapping."""
    matching_keys = []
    for hf_key, jax_p in hf_mapping.items():
        if re.match(hf_key, main_key):
            if jax_p is None:
                return None
            matching_keys.append(re.sub(hf_key, jax_p, main_key))

    if len(matching_keys) == 1:
        return matching_keys[0]

    raise TypeError(f"couldnt find key: {main_key}")


def build_hf_index(path, hf_mapping, n_layers) -> dict[tuple[str, ...], list[tuple[str, str]]]:
    index = {}
    files = list(glob.glob(path + "/*safetensors"))
    for file in files:
        with safe_open(file, framework="torch") as f:
            for hf_param_key in f.keys():  # noqa: SIM118
                jax_param_key = get_jax_key(hf_param_key, hf_mapping)
                if jax_param_key is None:
                    continue

                param_ending = jax_param_key.split(".")[-1]
                jax_path = jax_param_key.split(".")[0].split("/")

                layer = re.fullmatch(r"Block_([0-9]+)", jax_path[0])
                if layer:
                    jax_path[0] = "Block"

                jax_key = tuple(jax_path + [param_ending])
                location = (file, hf_param_key)

                if layer:
                    if index.get(jax_key) is None:
                        index[jax_key] = [None] * n_layers
                    index[jax_key][int(layer.group(1))] = location
                else:
                    index[jax_key] = [location]

    return index


def load_leaf(path, param, index: dict[tuple[str, ...], list[tuple[str, str]]]):
    keys = index[tuple(x.key for x in path)]
    transpose = path[-1].key == "kernel"
    stacked = path[0].key == "Block"

    def read(file, hf_key, idx):
        with safe_open(file, "pt") as f:
            s = f.get_slice(hf_key)
            x = s[idx[::-1]].T if transpose else s[idx]
        return x.float().numpy().astype(param.dtype)

    def cb(idx):
        if stacked:
            # TODO: if pipeline is added (layers are sharded, idx[0] != None) then have to index the layers
            # on this device/host
            return np.stack([read(f, k, idx[1:]) for f, k in keys], axis=0)
        file, hf_key = keys[0]
        return read(file, hf_key, idx)

    hf_loaded_param = jax.make_array_from_callback(param.shape, param.sharding, cb)
    return hf_loaded_param


def get_torch_weights_to_jax(params: PyTree, name: str, hf_mapping) -> PyTree:
    """Load Hugging Face model weights into a JAX PyTree of parameters."""
    path = os.path.join(HF_CHECKPOINT_PATH, name)
    download_hf_weights(name, path)

    n_layers = jax.tree.leaves(params["Block"])[0].shape[0]
    index = build_hf_index(path, hf_mapping, n_layers)
    params = jax.tree.map_with_path(partial(load_leaf, index=index), params)

    return params


def convert_weights(name: str, param: Array) -> torch.Tensor:
    """Convert JAX parameter to Hugging Face compatible tensor format."""

    # numpy can't store bfloat 16 convert to fp32 instead
    # this only applies to gamma param in qwen3
    if param.dtype == jnp.bfloat16:
        param = param.astype(jnp.float32)
    return torch.Tensor(param.T if "kernel" in name else param).contiguous()


def convert_to_jax_key(keys: tuple[DictKey]) -> str:
    """
    Convert key path from jax.tree.map_with_path to string format that matches the reverse HF mapping keys.

    Args:
        keys: A tuple of DictKeys representing the jax param path in the PyTree.
        Eg. (DictKey(key='Block_0'), DictKey(key='FeedForward_0'), DictKey(key='Dense_0'), DictKey(key='kernel'))
    Returns:
        A string representing the full key path.
        Eg. 'Block_0/FeedForward_0/Dense_0.kernel'
    """
    key_path = "/".join([k.key for k in keys[:-1]]) + f".{keys[-1].key}"
    return key_path


def convert_key(name: str, reverse_hf_mapping) -> str:
    """Convert JAX parameter key to Hugging Face parameter key using the reverse mapping."""
    matching_keys = []
    for jax_key, hf_p in reverse_hf_mapping.items():
        if re.match(jax_key, name):
            matching_keys.append(re.sub(jax_key, hf_p, name))

    if len(matching_keys) == 1:
        return matching_keys[0]

    raise TypeError(f"couldnt find key: {name}")


def convert_pytree(params: PyTree, reverse_hf_mapping) -> dict[str, torch.Tensor]:
    """Convert a JAX PyTree of parameters to a dictionary of Hugging Face compatible tensors."""
    loaded_tensors = {}

    def convert_param(key: tuple[DictKey], param: Array):
        jax_key = convert_to_jax_key(key)
        torch_key = convert_key(jax_key, reverse_hf_mapping)
        loaded_tensors[torch_key] = convert_weights(jax_key, param)

    jax.tree.map_with_path(convert_param, params)
    return loaded_tensors


def save_to_hf(dir_path: str, params: PyTree, hf_model_name: str, reverse_hf_mapping) -> None:
    hf_path = os.path.join(HF_CHECKPOINT_PATH, hf_model_name)
    download_hf_weights(hf_model_name, hf_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for file in os.listdir(hf_path):
        src_path = os.path.join(hf_path, file)
        if os.path.isfile(src_path) and not file.endswith(".safetensors"):
            shutil.copy(src_path, dir_path)

    new_tensors = convert_pytree(params, reverse_hf_mapping)
    save_file(new_tensors, f"{dir_path}/model.safetensors")


def get_lm_head_weights(params: PyTree, tie_weights: bool) -> Array:
    if tie_weights:
        return jnp.transpose(params["token_emb"]["embedding"])
    return params["Dense_0"]["kernel"]
