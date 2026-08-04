import jax.numpy as jnp
from jaxtyping import Array

from .layers import RopeCorrection
from .transformer import Transformer


def llama_rope_correction(freq: Array) -> Array:
    # from: https://github.com/jax-ml/jax-llm-examples/blob/main/llama3/llama3_jax/model.py
    factor = 8.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    old_context_len = 8192

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * jnp.pi / freq
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, freq / factor, freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


class Llama3(Transformer):
    qk_norm: bool = False
    rms_eps: float = 1e-5
    rope_correction: RopeCorrection = llama_rope_correction

    @property
    def hf_mapping(self):
        return {
            # embedding
            r"model\.embed_tokens\.weight": "token_emb.embedding",
            #  attention
            r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": r"Block_\1/GroupedQueryAttention_0/Dense_0.kernel",
            r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": r"Block_\1/GroupedQueryAttention_0/Dense_1.kernel",
            r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": r"Block_\1/GroupedQueryAttention_0/Dense_2.kernel",
            r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": r"Block_\1/GroupedQueryAttention_0/Dense_3.kernel",
            # mlp
            r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": r"Block_\1/FeedForward_0/Dense_0.kernel",
            r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": r"Block_\1/FeedForward_0/Dense_1.kernel",
            r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": r"Block_\1/FeedForward_0/Dense_2.kernel",
            # block norms
            r"model\.layers\.([0-9]+)\.input_layernorm\.weight": r"Block_\1/RMSNorm_0.gamma",
            r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"Block_\1/RMSNorm_1.gamma",
            # final rms
            r"model\.norm\.weight": "RMSNorm_0.gamma",
            r"lm_head\.weight": "Dense_0.kernel",
        }

    @property
    def reverse_hf_mapping(self):
        return {
            r"token_emb\.embedding": r"model.embed_tokens.weight",
            # block norms
            r"Block_([0-9]+)/RMSNorm_0\.gamma": r"model.layers.\1.input_layernorm.weight",
            r"Block_([0-9]+)/RMSNorm_1\.gamma": r"model.layers.\1.post_attention_layernorm.weight",
            # gqa projections
            r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_0\.kernel": r"model.layers.\1.self_attn.q_proj.weight",
            r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_1\.kernel": r"model.layers.\1.self_attn.k_proj.weight",
            r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_2\.kernel": r"model.layers.\1.self_attn.v_proj.weight",
            r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_3\.kernel": r"model.layers.\1.self_attn.o_proj.weight",
            # mlp
            r"Block_([0-9]+)/FeedForward_0/Dense_0\.kernel": r"model.layers.\1.mlp.gate_proj.weight",
            r"Block_([0-9]+)/FeedForward_0/Dense_1\.kernel": r"model.layers.\1.mlp.up_proj.weight",
            r"Block_([0-9]+)/FeedForward_0/Dense_2\.kernel": r"model.layers.\1.mlp.down_proj.weight",
            # final rms
            r"RMSNorm_0\.gamma": r"model.norm.weight",
            r"Dense_0\.kernel": r"lm_head.weight",
        }
