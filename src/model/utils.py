import functools
import glob
import os
import re
import shutil

import jax
import jax.numpy as jnp
import torch
from huggingface_hub import snapshot_download
from jax.sharding import PartitionSpec as P
from jax.tree_util import DictKey
from jaxtyping import Array, PyTree
from safetensors import safe_open
from safetensors.torch import save_file
from stax.sharding.main import AXIS_NAMES_ENUM

DP = AXIS_NAMES_ENUM.DP.value
FSDP = AXIS_NAMES_ENUM.FSDP.value
CP_ULYSSES = AXIS_NAMES_ENUM.CP_ULYSSES.value


def convert_dtype(dtype_str: str) -> jnp.dtype:
    match dtype_str:
        case "float32":
            return jnp.float32
        case "bfloat16":
            return jnp.bfloat16
        case "float16":
            return jnp.float16
        case _:
            raise ValueError(f"Unsupported dtype string: {dtype_str}")


def dynamic_slice_rows(x: Array, start: Array, size: int) -> Array:
    rows = jnp.arange(x.shape[0])[:, None]
    cols = start[:, None] + jnp.arange(size)[None, :]
    return x[rows, cols]


def dynamic_update_rows(x: Array, update: Array, start: Array) -> Array:
    rows = jnp.arange(x.shape[0])[:, None]
    cols = start[:, None] + jnp.arange(update.shape[1])[None, :]
    return x.at[rows, cols].set(update)


def make_prompt_mask(kv_len: int, cache_len: Array, seq_lens: Array) -> Array:
    raw_length = jnp.arange(kv_len)[None, :]  # [1, max_seq]
    cache_len = cache_len[:, None]  # [B, 1]
    left_padding_mask = raw_length >= (cache_len - seq_lens[:, None])  # [B, max_seq]
    valid_cache_mask = raw_length < cache_len
    return left_padding_mask & valid_cache_mask  # [B, mask_seq]


def make_tril_mask(query_length: int, key_length: int, t_start: Array) -> Array:
    q = jnp.arange(query_length)[None, :, None] + t_start[:, None, None]  # [B, 1, 1]
    k = jnp.arange(key_length)[None, None, :]  # [B, 1, 1]
    return q >= k


def make_attention_mask(query_length: int, key_length: int, t_start: Array, seq_lens: Array) -> Array:
    prompt_mask = make_prompt_mask(key_length, t_start + query_length, seq_lens)  # B, key_shape
    tril = make_tril_mask(query_length, key_length, t_start)[:, None, :, :]  # B, 1, query_shape, key_shape
    return prompt_mask[:, None, None, :] * tril


def download_hf_weights(name: str):
    if not os.path.isdir(name):
        snapshot_download(
            repo_id=name,
            local_dir=name,
            token=os.environ.get("HF_TOKEN", None),
        )


def delete_hf_weights(name: str):
    if os.path.isdir(name):
        shutil.rmtree(name)


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
    download_hf_weights(name)
    torch_hf_params = {}

    files = list(glob.glob(name + "/*safetensors"))
    for file in files:
        with safe_open(file, framework="torch") as f:
            for hf_param_key in f.keys():
                torch_hf_params[hf_param_key] = f.get_tensor(hf_param_key)
                jax_param_key = get_jax_key(hf_param_key, hf_mapping)

                if jax_param_key is None:
                    continue

                param_ending = jax_param_key.split(".")[-1]
                jax_path = jax_param_key.split(".")[0].split("/")

                jax_param = params

                for node in jax_path:
                    jax_param = jax_param[node]

                new_param = torch_hf_params[hf_param_key].float()
                new_param = new_param.T.numpy() if "kernel" in param_ending else new_param.numpy()

                assert new_param.shape == jax_param[param_ending].shape, (
                    f"Shape mismatch for {jax_param_key}: expected {jax_param[param_ending].shape}, got {new_param.shape}"
                )
                jax_param[param_ending] = new_param

    delete_hf_weights(name)
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
    download_hf_weights(hf_model_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for file in os.listdir(hf_model_name):
        src_path = os.path.join(hf_model_name, file)
        if os.path.isfile(src_path) and not file.endswith(".safetensors"):
            shutil.copy(src_path, dir_path)

    new_tensors = convert_pytree(params, reverse_hf_mapping)
    save_file(new_tensors, f"{dir_path}/model.safetensors")


def get_embedding_weights(params: PyTree) -> Array:
    return jnp.transpose(params["token_emb"]["embedding"])


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
    chunked_hidden_inputs = jax.lax.with_sharding_constraint(
        h.reshape(num_chunks, chunk_size, D), P(None, (DP, FSDP, CP_ULYSSES), None)
    )  # num_chunks, chunk_size, D
    chunked_targets = jax.lax.with_sharding_constraint(
        targets.reshape(num_chunks, chunk_size), P(None, (DP, FSDP, CP_ULYSSES))
    )  # num_chunks, chunk_size

    W_gather = jax.lax.with_sharding_constraint(W, P())

    def body_fn(carry, chunk):
        hidden_chunk, target_chunk = chunk
        logits_chunked = hidden_chunk @ W_gather  # chunk_size x V

        chunk_max = jnp.max(logits_chunked, axis=-1)  # chunk_size
        shifted_logits = logits_chunked - chunk_max[:, None]  # chunk_size x V

        exp_shifted = jnp.exp(shifted_logits)
        sum_exp = jnp.sum(exp_shifted, -1)[:, None]  # chunk_size x 1
        log_softmax = shifted_logits - jnp.log(sum_exp)  # chunk_size x V

        target_log_probs = jnp.take_along_axis(log_softmax, target_chunk[:, None], axis=-1)[:, 0]  # chunk_size

        return carry, target_log_probs

    _, chunked_output = jax.lax.scan(body_fn, None, (chunked_hidden_inputs, chunked_targets))  # num_chunks x chunk_size

    output = chunked_output.reshape(B)  # (B,)

    return output, (chunked_hidden_inputs, W_gather, chunked_targets)


def _bwd(chunk_size, residuals, dy):
    chunked_hidden_inputs, W_gather, chunked_targets = residuals
    num_chunks, chunk_size, D = chunked_hidden_inputs.shape
    _, V = W_gather.shape

    chunked_dy = jax.lax.with_sharding_constraint(
        dy.astype(jnp.float32).reshape(num_chunks, chunk_size), P(None, (DP, FSDP, CP_ULYSSES))
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
