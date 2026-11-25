from typing import cast

import jax.numpy as jnp
from jaxtyping import Array
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .layers import RopeCorrection
from .layers.transformer import Transformer

# from https://huggingface.co/allenai/Olmo-3-1025-7B/blob/main/config.json
YARN_FACTOR = 8.0
YARN_ORIGINAL_MAX_LEN = 8192
YARN_BETA_FAST = 32.0
YARN_BETA_SLOW = 1.0
YARN_ATTENTION_FACTOR = 1.2079441541679836


def yarn_rope_correction(freq: Array) -> Array:
    # from: transformers/modeling_rope_utils.py `_compute_yarn_parameters`
    n = freq.shape[0]
    log_step = -jnp.log(freq[1])  # freq[i] = base ** (-2i / head_dim), so this is 2 * log(base) / head_dim

    def correction_dim(num_rotations: float) -> Array:
        return jnp.log(YARN_ORIGINAL_MAX_LEN / (num_rotations * 2 * jnp.pi)) / log_step

    low = jnp.maximum(jnp.floor(correction_dim(YARN_BETA_FAST)), 0)
    high = jnp.minimum(jnp.ceil(correction_dim(YARN_BETA_SLOW)), 2 * n - 1)

    # 0 keeps the original frequency, 1 divides it by YARN_FACTOR
    ramp = jnp.clip((jnp.arange(n, dtype=jnp.float32) - low) / (high - low), 0.0, 1.0)
    return (freq / YARN_FACTOR) * ramp + freq * (1 - ramp)


class Olmo3(Transformer):
    qk_norm: bool = True
    qk_norm_per_head: bool = False
    post_norm: bool = True
    rms_eps: float = 1e-6
    rope_correction: RopeCorrection = yarn_rope_correction
    rope_scale: float = YARN_ATTENTION_FACTOR
    sliding_window: int | None = 4096
    full_attention_every: int = 4

    @staticmethod
    def eos_token_id(tokenizer: PreTrainedTokenizerBase) -> int | None:
        # a chat turn ends with <|im_end|>, the tokenizer's own eos is <|endoftext|>
        return cast(int | None, tokenizer.convert_tokens_to_ids("<|im_end|>"))

    @property
    def hf_mapping(self):
        return {
            # embedding
            r"model\.embed_tokens\.weight": "token_emb.embedding",
            # block norms (post norm: applied to the attention / mlp output)
            r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"Block_\1/RMSNorm_0.gamma",
            r"model\.layers\.([0-9]+)\.post_feedforward_layernorm\.weight": r"Block_\1/RMSNorm_1.gamma",
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

    @property
    def reverse_hf_mapping(self):
        return {
            # embedding
            r"token_emb\.embedding": r"model.embed_tokens.weight",
            # block norms
            r"Block_([0-9]+)/RMSNorm_0\.gamma": r"model.layers.\1.post_attention_layernorm.weight",
            r"Block_([0-9]+)/RMSNorm_1\.gamma": r"model.layers.\1.post_feedforward_layernorm.weight",
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
