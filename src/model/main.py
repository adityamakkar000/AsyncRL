import glob
import os
import re
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from huggingface_hub import snapshot_download
from jaxtyping import Array, PyTree
from optax import GradientTransformation
from safetensors import safe_open
from stax.model_module import HFModelBase

from src.model.qwen3 import KVCache, Qwen3


@dataclass
class config:
    hf_weight_download_dir: str = "hf_qwen3_0_6b"
    chosen_model_size: str = "0.6B"
    model_name: str = "Qwen/Qwen3-0.6B"
    batch_size: int = 3


class mainModel(HFModelBase):
    # TODO: change to DictConfig
    def __init__(self, config: config):
        self.config = config
        self.model = None
        super().__init__()

    def load_from_hf(self):
        download_dir = self.config.hf_weight_download_dir
        model_name = self.config.model_name
        snapshot_download(
            repo_id=model_name,
            local_dir=download_dir,
        )

    def return_qwen_config(self) -> dict[str, int | jnp.dtype]:
        model_size = self.config.chosen_model_size
        if model_size == "0.6B":
            QWEN3_CONFIG = {
                "vocab_size": 151936,
                "d_ff": 3072,
                "sequence_len": 8192,
                "model_dim": 1024,
                "n_heads": 16,
                "n_groups": 8,
                "n_layers": 28,
                "head_dim": 128,
                "model_dtype": jnp.float32,
            }

        return QWEN3_CONFIG

    def return_HF_mapping(self) -> dict[str, str]:
        return {  # embedding
            r"model\.embed_tokens\.weight": "token_emb.embedding",
            # block norms
            r"model\.layers\.([0-9]+)\.input_layernorm\.weight": r"Block_\1/RMSNorm_0.gamma",  # 1024
            r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"Block_\1/RMSNorm_1.gamma",  # 1024
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

    def get_jax_key(self, main_key: str) -> str | None:
        HF_MAPPING = self.return_HF_mapping()
        matching_keys = []
        for hf_key, jax_p in HF_MAPPING.items():
            if re.match(hf_key, main_key):
                matching_keys.append(re.sub(hf_key, jax_p, main_key))

        if len(matching_keys) == 1:
            return matching_keys[0]

        raise TypeError(f"couldnt find key: {main_key}")

    def update_weights(self, params: PyTree) -> PyTree:
        download_dir = self.config.hf_weight_download_dir

        torch_hf_params = {}

        files = list(glob.glob(download_dir + "/*safetensors"))
        for file in files:
            with safe_open(file, framework="torch") as f:
                for hf_param_key in f.keys():
                    torch_hf_params[hf_param_key] = f.get_tensor(hf_param_key)
                    jax_param_key = self.get_jax_key(hf_param_key)

                    if jax_param_key is None:
                        raise TypeError("Could not find matching JAX key.")

                    param_ending = jax_param_key.split(".")[-1]
                    jax_path = jax_param_key.split(".")[0].split("/")

                    jax_param = params["params"]

                    for node in jax_path:
                        jax_param = jax_param[node]

                    if "kernel" in param_ending:
                        new_param = torch_hf_params[hf_param_key].float().T.numpy()
                    else:
                        new_param = torch_hf_params[hf_param_key].float().numpy()

                    assert new_param.shape == jax_param[param_ending].shape
                    jax_param[param_ending] = new_param

        return params

    def init_state(
        self, chosen_model: str, rng: Array, tx: Optional[GradientTransformation]
    ) -> tuple[PyTree, PyTree | None, Qwen3]:
        qwen_3_config = self.return_qwen_config()

        if not os.path.isdir(self.config.hf_weight_download_dir):
            self.load_from_hf()

        model = Qwen3(
            vocab_size=qwen_3_config["vocab_size"],
            d_ff=qwen_3_config["d_ff"],
            sequence_len=qwen_3_config["sequence_len"],
            model_dim=qwen_3_config["model_dim"],
            n_heads=qwen_3_config["n_heads"],
            n_groups=qwen_3_config["n_groups"],
            head_dim=qwen_3_config["head_dim"],
            n_layers=qwen_3_config["n_layers"],
            model_dtype=qwen_3_config["model_dtype"],
        )

        x = jnp.ones((1, 1), dtype=jnp.int32)
        seq_lens = jnp.array([1])
        initial_cache = self.init_kv_cache()

        params = model.init(rng, x, seq_lens, initial_cache)
        optax_state = tx.init(params) if tx is not None else None

        qwen3_params = self.update_weights(params)
        self.model = model

        return qwen3_params, optax_state, model

    def load_to_hf():
        # TODO: implement method to load model weights to huggingface
        pass

    def init_kv_cache(self) -> list[KVCache]:
        qwen_config = self.return_qwen_config()
        B = self.config.batch_size
        n_layers = qwen_config["n_layers"]
        n_groups = qwen_config["n_groups"]
        max_sequence_len = qwen_config["sequence_len"]
        head_dim = qwen_config["head_dim"]

        initial_cache: list[KVCache] = []
        for _ in range(n_layers):
            length = 0
            k = jnp.zeros((B, n_groups, max_sequence_len, head_dim), dtype=jnp.bfloat16)
            v = jnp.zeros((B, n_groups, max_sequence_len, head_dim), dtype=jnp.bfloat16)
            _cache = KVCache(
                k=k,
                v=v,
                length=length,
            )
            initial_cache.append(_cache)

        return initial_cache

    def __call__(
        self,
        *,
        model: Qwen3,
        x: Array,
        params: Array,
        kv_cache: list[KVCache],
        tx: Optional[GradientTransformation] = None,
        sequence_lens: Array,
    ) -> Array:
        logits, cache = model.apply(params, x, sequence_lens, kv_cache)

        return logits, cache
