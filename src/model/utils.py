import glob
import os
import re
import shutil

import jax
import jax.numpy as jnp
import torch
from huggingface_hub import snapshot_download
from jax.tree_util import DictKey
from jaxtyping import Array, PyTree
from safetensors import safe_open
from safetensors.torch import save_file


def convert_dtype(dtype_str: str) -> jnp.dtype:
    """Convert a string representation of a data type to a JAX data type."""
    match dtype_str:
        case "float32":
            return jnp.float32
        case "bfloat16":
            return jnp.bfloat16
        case "float16":
            return jnp.float16
        case _:
            raise ValueError(f"Unsupported dtype string: {dtype_str}")


def make_prompt_mask(max_seq_len: int, cache_len, seq_lens: Array) -> Array:
    """
    This function generates a boolean mask that identifies valid (non-padded) tokens
    within the cache region of each sequence. It handles left-padded sequences by
    masking out padding tokens at the beginning of each sequence.

        max_seq_len (int): The maximum sequence length including any tokens beyond the cache.
        cache_len (int): The length of the cached tokens (KV cache size).
        seq_lens (Array): Array of shape (batch_size,) containing the actual
            sequence lengths for each batch element. Each element should be <= cache_len.

    Returns:
        Array: A boolean mask of shape (batch_size, max_seq_len) where True indicates
            valid (non-padded) tokens within the cache region, and False indicates either
            padding tokens or positions beyond the cache.

    Example:
        >>> seq_lens = jnp.array([3, 5])
        >>> max_seq_len = 5
        >>> cache_len = 4
        >>> make_prompt_mask(max_seq_len, cache_len, seq_lens)
        # Returns:
        # [[False, False, True, True, False],
        #  [True,  True,  True, True, False]]
        #
        # First sequence: 3 valid tokens, left-padded with 1 token, 1 position beyond cache
        # Second sequence: 4 valid tokens (capped by cache_len), 1 position beyond cache
    """
    raw_length = jnp.arange(max_seq_len)[None, :]
    # left padding mask
    padding_mask = raw_length >= (cache_len - seq_lens[:, None])
    # cache mask
    cache_mask = raw_length < cache_len
    return padding_mask & cache_mask


def make_tril_mask(query_shape: int, key_shape: int, t_start: int) -> Array:
    """
    Create a lower triangular mask for attention mechanisms.
    This function generates a boolean mask where each query position can only
    attend to key positions that are at or before its temporal position,
    adjusted by a starting offset.
    Args:
        query_shape: The size of the query dimension (number of query positions).
        key_shape: The size of the key dimension (number of key positions).
        t_start: The temporal offset to apply to query positions.
    Returns:
        Array: A boolean array of shape (query_shape, key_shape) where
            True indicates the query position can attend to the key position
            (i.e., query_position + t_start >= key_position).
    Example:
        >>> t = 3
        >>> T = 5
        >>> tril_mask = make_tril_mask(t, T, t_start=0)
        # Returns:
        # [[ True, False, False, False, False],
        #  [ True,  True, False, False, False],
        #  [ True,  True,  True, False, False]]
        # Each query position can attend to all key positions up to its own index.
        >>> tril_mask = make_tril_mask(t, T, t_start=2)
        # Returns:
        # [[True, True, True, False, False],
        #  [ True, True, True, True, False],
        #  [ True,  True, True, True, True]]
    """

    return (jnp.arange(query_shape)[:, None] + t_start) >= (jnp.arange(key_shape)[None, :])


def make_attention_mask(query_shape: int, key_shape: int, t_start: int, seq_lens: Array) -> Array:
    """
    Create an attention mask for sequences with padding and causal masking.
    Args:
        query_shape (int): Current time step or query length.
        key_shape (int): Total sequence length or key length.
        t_start (int): The temporal offset to apply to query positions.
        seq_lens (Array): Array of sequence lengths for each batch element.
    Returns:
        Array: Attention mask of shape (batch_size, 1, query_shape, key_shape).

    example:
        seq_lens = jnp.array([3, 5])
        query_shape = 3
        key_shape = 5
        prompt_mask = [[0 0 1 1 1]
                       [1 1 1 1 1]]
        tril = [[[True True True False False],
                 [True True True True False],
                 [True True True True True]]]
        returns:
        [[[0 0 1 0 0]
          [0 0 1 1 0]
          [0 0 1 1 1]]

         [[1 1 1 0 0]
          [1 1 1 1 0]
          [1 1 1 1 1]]]
    """

    prompt_mask = make_prompt_mask(key_shape, t_start + query_shape, seq_lens)  # B, key_shape
    tril = make_tril_mask(query_shape, key_shape, t_start)[None, None, :, :]  # 1,1, query_shape, key_shape
    return prompt_mask[:, None, None, :] * tril


HF_MAPPING = {  # embedding
    r"model\.embed_tokens\.weight": "token_emb.embedding",
    # block norms
    r"model\.layers\.([0-9]+)\.input_layernorm\.weight": r"Block_\1/RMSNorm_0.gamma",
    r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"Block_\1/RMSNorm_1.gamma",
    # gqa
    r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": r"Block_\1/GroupedQueryAttention_0/Dense_0.kernel",
    r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"Block_\1/GroupedQueryAttention_0/Dense_1.kernel",
    r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": r"Block_\1/GroupedQueryAttention_0/Dense_2.kernel",
    r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": r"Block_\1/GroupedQueryAttention_0/Dense_3.kernel",
    # gqa norms
    r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": r"Block_\1/GroupedQueryAttention_0/RMSNorm_0.gamma",
    r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": r"Block_\1/GroupedQueryAttention_0/RMSNorm_1.gamma",
    # mlp
    r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": r"Block_\1/FeedForward_0/Dense_0.kernel",
    r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": r"Block_\1/FeedForward_0/Dense_1.kernel",
    r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": r"Block_\1/FeedForward_0/Dense_2.kernel",
    # final rms
    r"model\.norm\.weight": "RMSNorm_0.gamma",
    r"lm_head\.weight": "Dense_0.kernel",
}

REVERSE_HF_MAPPING = {
    # embedding
    r"token_emb\.embedding": r"model.embed_tokens.weight",
    # block norms
    r"Block_([0-9]+)/RMSNorm_0\.gamma": r"model.layers.\1.input_layernorm.weight",
    r"Block_([0-9]+)/RMSNorm_1\.gamma": r"model.layers.\1.post_attention_layernorm.weight",
    # gqa projections
    r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_0\.kernel": r"model.layers.\1.self_attn.q_proj.weight",
    r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_1\.kernel": r"model.layers.\1.self_attn.k_proj.weight",
    r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_2\.kernel": r"model.layers.\1.self_attn.v_proj.weight",
    r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_3\.kernel": r"model.layers.\1.self_attn.o_proj.weight",
    # gqa norms
    r"Block_([0-9]+)/GroupedQueryAttention_0/RMSNorm_0\.gamma": r"model.layers.\1.self_attn.q_norm.weight",
    r"Block_([0-9]+)/GroupedQueryAttention_0/RMSNorm_1\.gamma": r"model.layers.\1.self_attn.k_norm.weight",
    # mlp
    r"Block_([0-9]+)/FeedForward_0/Dense_0\.kernel": r"model.layers.\1.mlp.gate_proj.weight",
    r"Block_([0-9]+)/FeedForward_0/Dense_1\.kernel": r"model.layers.\1.mlp.up_proj.weight",
    r"Block_([0-9]+)/FeedForward_0/Dense_2\.kernel": r"model.layers.\1.mlp.down_proj.weight",
    # final rms + lm head
    r"RMSNorm_0\.gamma": r"model.norm.weight",
    r"Dense_0\.kernel": r"lm_head.weight",
}


def download_hf_weights(name: str):
    """Download model weights from Hugging Face if not already present locally."""
    if not os.path.isdir(name):
        snapshot_download(
            repo_id=name,
            local_dir=name,
            token=os.environ.get("HF_TOKEN", None),
        )


def get_jax_key(main_key: str) -> str | None:
    """Convert Hugging Face parameter key to JAX parameter key using the mapping."""
    matching_keys = []
    for hf_key, jax_p in HF_MAPPING.items():
        if re.match(hf_key, main_key):
            matching_keys.append(re.sub(hf_key, jax_p, main_key))

    if len(matching_keys) == 1:
        return matching_keys[0]

    raise TypeError(f"couldnt find key: {main_key}")


def get_qwen_3_weights(params: PyTree, name: str) -> PyTree:
    """Load Hugging Face model weights into a JAX PyTree of parameters."""
    download_hf_weights(name)
    torch_hf_params = {}

    files = list(glob.glob(name + "/*safetensors"))
    for file in files:
        with safe_open(file, framework="torch") as f:
            for hf_param_key in f.keys():
                torch_hf_params[hf_param_key] = f.get_tensor(hf_param_key)
                jax_param_key = get_jax_key(hf_param_key)

                if jax_param_key is None:
                    raise TypeError("Could not find matching JAX key.")

                param_ending = jax_param_key.split(".")[-1]
                jax_path = jax_param_key.split(".")[0].split("/")

                jax_param = params

                for node in jax_path:
                    jax_param = jax_param[node]

                new_param = torch_hf_params[hf_param_key].float()
                new_param = new_param.T.numpy() if "kernel" in param_ending else new_param.numpy()

                assert new_param.shape == jax_param[param_ending].shape
                jax_param[param_ending] = new_param

    return params


def convert_weights(name: str, param: Array) -> torch.Tensor:
    """Convert JAX parameter to Hugging Face compatible tensor format."""
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


def convert_key(name: str) -> str:
    """Convert JAX parameter key to Hugging Face parameter key using the reverse mapping."""
    matching_keys = []
    for jax_key, hf_p in REVERSE_HF_MAPPING.items():
        if re.match(jax_key, name):
            matching_keys.append(re.sub(jax_key, hf_p, name))

    if len(matching_keys) == 1:
        return matching_keys[0]

    raise TypeError(f"couldnt find key: {name}")


def convert_pytree(params: PyTree) -> dict[str, torch.Tensor]:
    """Convert a JAX PyTree of parameters to a dictionary of Hugging Face compatible tensors."""
    loaded_tensors = {}

    def convert_param(key: tuple[DictKey], param: Array):
        jax_key = convert_to_jax_key(key)
        torch_key = convert_key(jax_key)
        loaded_tensors[torch_key] = convert_weights(jax_key, param)

    jax.tree.map_with_path(convert_param, params)
    return loaded_tensors


def save_to_hf(dir_path: str, params: PyTree, hf_model_name: str) -> None:
    download_hf_weights(hf_model_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for file in os.listdir(hf_model_name):
        src_path = os.path.join(hf_model_name, file)
        if os.path.isfile(src_path) and not file.endswith(".safetensors"):
            shutil.copy(src_path, dir_path)

    new_tensors = convert_pytree(params)
    save_file(new_tensors, f"{dir_path}/model.safetensors")
