import functools
import glob
import os
import re
import shutil
from collections.abc import Callable

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
from stax.sharding.main import AXIS_NAMES_ENUM

from src.constants import HF_CHECKPOINT_PATH

DP = AXIS_NAMES_ENUM.DP.value
FSDP = AXIS_NAMES_ENUM.FSDP.value
CP_ULYSSES = AXIS_NAMES_ENUM.CP_ULYSSES.value

ACT_SPEC = P((DP, FSDP), CP_ULYSSES, None)
QKV_SPEC = P((DP, FSDP), CP_ULYSSES, None, None)
OUT_SPEC = P((DP, FSDP), None, CP_ULYSSES, None)


def act_constrain(x: PyTree[Array], shard: bool = True, spec: P = ACT_SPEC) -> PyTree[Array]:
    return jax.lax.with_sharding_constraint(x, spec) if shard else x


def quantize(x: Array) -> tuple[Array, Array]:
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale = (amax / 127.0 + jnp.finfo(x.dtype).tiny).astype(jnp.bfloat16)
    q = jnp.clip(jnp.round(x / scale), -127, 127).astype(jnp.int8)
    return q, scale.squeeze(axis=-1)


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


def get_torch_weights_to_jax(params: PyTree, name: str, hf_mapping) -> PyTree:
    """Load Hugging Face model weights into a JAX PyTree of parameters."""
    path = os.path.join(HF_CHECKPOINT_PATH, name)
    download_hf_weights(name, path)
    torch_hf_params = {}

    files = list(glob.glob(path + "/*safetensors"))
    for file in files:
        with safe_open(file, framework="torch") as f:
            for hf_param_key in f.keys():  # noqa: SIM118
                torch_hf_params[hf_param_key] = f.get_tensor(hf_param_key)
                jax_param_key = get_jax_key(hf_param_key, hf_mapping)

                if jax_param_key is None:
                    continue

                param_ending = jax_param_key.split(".")[-1]
                jax_path = jax_param_key.split(".")[0].split("/")

                layer = re.fullmatch(r"Block_([0-9]+)", jax_path[0])
                if layer:
                    jax_path[0] = "Block"

                jax_param = params

                for node in jax_path:
                    jax_param = jax_param[node]

                new_param = torch_hf_params.pop(hf_param_key).float()
                new_param = new_param.T.numpy() if "kernel" in param_ending else new_param.numpy()

                target = jax_param[param_ending]
                expected = target.shape[1:] if layer else target.shape
                assert new_param.shape == expected, (
                    f"Shape mismatch for {jax_param_key}: expected {expected}, got {new_param.shape}"
                )

                if layer:
                    if not isinstance(target, np.ndarray):
                        target = jax_param[param_ending] = np.empty(target.shape, dtype=new_param.dtype)
                    target[int(layer.group(1))] = new_param
                else:
                    jax_param[param_ending] = new_param

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


def make_weight_shard_fn(weight_shard_dim: int = 0) -> Callable[[str, Array], int | None]:
    def weight_shard_fn(path: str, param: Array) -> int | None:
        if param.ndim < 2:
            return None
        if re.search(r"\['Block'\]", path):
            return weight_shard_dim + 1 if param.ndim >= 3 else None
        return weight_shard_dim

    return weight_shard_fn


def get_lm_head_weights(params: PyTree, tie_weights: bool) -> Array:
    if tie_weights:
        return jnp.transpose(params["token_emb"]["embedding"])
    return params["Dense_0"]["kernel"]


def make_chunks(B: int, chunk_size: int):
    assert B % chunk_size == 0, "B must be divisible by chunk_size"
    num_chunks = B // chunk_size
    return num_chunks


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def fused_linear_selection(h, W, targets, chunk_size=1024) -> Array:
    return _fwd(h, W, targets, chunk_size)[0]


def _fwd(h, W, targets, chunk_size):
    B, D = h.shape
    _, _V = W.shape
    chunk_size = min(chunk_size, B)

    num_chunks = make_chunks(B, chunk_size)
    chunked_hidden_inputs = act_constrain(
        h.reshape(num_chunks, chunk_size, D), spec=P(None, (DP, FSDP, CP_ULYSSES), None)
    )  # num_chunks, chunk_size, D
    chunked_targets = act_constrain(
        targets.reshape(num_chunks, chunk_size), spec=P(None, (DP, FSDP, CP_ULYSSES))
    )  # num_chunks, chunk_size

    W_gather = act_constrain(W, spec=P())

    chunked_hidden_inputs, W_gather = jax.tree.map(lambda x: x.astype(jnp.float32), (chunked_hidden_inputs, W_gather))

    def body_fn(carry, chunk):
        hidden_chunk, target_chunk = chunk
        logits_chunked = hidden_chunk @ W_gather  # chunk_size x V
        log_softmax = jax.nn.log_softmax(logits_chunked, axis=-1)  # chunk_size x V
        target_log_probs = jnp.take_along_axis(log_softmax, target_chunk[:, None], axis=-1)[:, 0]  # chunk_size

        return carry, target_log_probs

    _, chunked_output = jax.lax.scan(body_fn, None, (chunked_hidden_inputs, chunked_targets))  # num_chunks x chunk_size

    output = chunked_output.reshape(B)  # (B,)

    return output, (chunked_hidden_inputs, W_gather, chunked_targets)


def _bwd(chunk_size, residuals, dy):
    chunked_hidden_inputs, W_gather, chunked_targets = residuals
    num_chunks, chunk_size, D = chunked_hidden_inputs.shape
    _, V = W_gather.shape

    chunked_dy = act_constrain(
        dy.astype(jnp.float32).reshape(num_chunks, chunk_size), spec=P(None, (DP, FSDP, CP_ULYSSES))
    )

    def body(dW, chunk):
        hidden_chunk, target_chunk, dy_chunk = chunk
        logits_chunked = hidden_chunk @ W_gather  # chunk_size x V

        p = jax.nn.softmax(logits_chunked, axis=-1)  # chunk_size x V
        onehot = jax.nn.one_hot(target_chunk, V, dtype=p.dtype)
        dz = dy_chunk[:, None] * (onehot - p)  # chunk_size x V

        dh_chunk = dz @ W_gather.T  # chunk_size x D
        dW = dW + hidden_chunk.T @ dz  # D x V
        return dW, dh_chunk

    dW, chunked_dh = jax.lax.scan(
        body,
        jnp.zeros_like(W_gather, dtype=jnp.float32),
        (chunked_hidden_inputs, chunked_targets, chunked_dy),
    )

    dh = chunked_dh.reshape(-1, D)  # (B, D)
    d_targets = jnp.zeros((num_chunks * chunk_size), jax.dtypes.float0)
    return dh, dW, d_targets


fused_linear_selection.defvjp(_fwd, _bwd)
