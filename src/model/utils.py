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
    if dtype_str == "float32":
        return jnp.float32
    elif dtype_str == "bfloat16":
        return jnp.bfloat16
    elif dtype_str == "float16":
        return jnp.float16
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")


def make_prompt_mask(padding_len: int, seq_lens: jax.Array) -> jax.Array:
    """
    Create a prompt mask to handle left-padding in sequences.
    Args:
        padding_len (int): The total length of the sequences including padding.
        seq_lens (jax.Array): Array of sequence lengths for each batch element.
    Returns:
        jax.Array: Prompt mask of shape (batch_size, padding_len).

    example:
        seq_lens = jnp.array([3, 5])
        padding_len = 5
        [[0 1 2 3 4]] >= [[2], [0]]
        returns:
        [[0 0 1 1 1]
            [1 1 1 1 1]]
    """

    return jnp.arange(padding_len)[None, :] >= (padding_len - seq_lens[:, None])


def make_tril_mask(t: int, T: int) -> jax.Array:
    """
    Create a lower triangular mask for causal attention.
    Args:
        t (int): Current time step or query length.
        T (int): Total sequence length or key length.
    Returns:
        jax.Array: Lower triangular mask of shape (t, T) [last row will always be True].

    example:
        t = 3
        T = 5

        T_arange = [[0 1 2 3 4]]
        query_position = [[0 1 2 3 4],
                            [0 1 2 3 4],
                            [0 1 2 3 4]]
        key_position  = [[ 0 0 0 0 0],
                            [ 1 1 1 1 1],
                            [ 2 2 2 2 2],
                            [ 3 3 3 3 3],
                            [ 4 4 4 4 4]]
        key_position[-t:] = [[2 2 2 2 2],
                                [3 3 3 3 3],
                                [4 4 4 4 4]]
        returns:
        [[True True True False False],
            [True True True True False],
            [True True True True True]]

    """

    T_arange = jnp.arange(T)[None, :]
    query_position = jnp.repeat(T_arange, t, axis=0)
    key_position = jnp.transpose(jnp.repeat(T_arange, T, axis=0))
    return query_position <= key_position[-t:, :]


def make_attention_mask(t: int, T: int, seq_lens: jax.Array) -> jax.Array:
    """
    Create an attention mask for sequences with padding and causal masking.
    Args:
        t (int): Current time step or query length.
        T (int): Total sequence length or key length.
        seq_lens (jax.Array): Array of sequence lengths for each batch element.
    Returns:
        jax.Array: Attention mask of shape (batch_size, 1, t, T).

    example:
        seq_lens = jnp.array([3, 5])
        t = 3
        T = 5
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

    prompt_mask = make_prompt_mask(padding_len=T, seq_lens=seq_lens)
    tril = make_tril_mask(t, T)[None, None, :, :]  # 1,1, t, T
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
    if not os.path.isdir(name):
        snapshot_download(
            repo_id=name,
            local_dir=name,
            token=os.environ.get("HF_TOKEN", None),
        )


def get_jax_key(main_key: str) -> str | None:
    matching_keys = []
    for hf_key, jax_p in HF_MAPPING.items():
        if re.match(hf_key, main_key):
            matching_keys.append(re.sub(hf_key, jax_p, main_key))

    if len(matching_keys) == 1:
        return matching_keys[0]

    raise TypeError(f"couldnt find key: {main_key}")


def get_qwen_3_weights(params: PyTree, name: str) -> PyTree:
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
