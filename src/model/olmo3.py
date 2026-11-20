import jax.numpy as jnp
from jaxtyping import Array

from .layers import RopeCorrection
from .transformer import Transformer

# rope_scaling block of the Olmo 3 config.json (rope_type: yarn)
YARN_FACTOR = 8.0
YARN_ORIGINAL_MAX_LEN = 8192
YARN_BETA_FAST = 32.0
YARN_BETA_SLOW = 1.0
YARN_ATTENTION_FACTOR = 1.2079441541679836

# every FULL_ATTENTION_EVERY-th layer is full attention, the rest only attend to the last SLIDING_WINDOW tokens
FULL_ATTENTION_EVERY = 4
SLIDING_WINDOW = 4096


def yarn_rope_correction(freq: Array) -> Array:
    # from: transformers.modeling_rope_utils._compute_yarn_parameters (truncate=True)
    n = freq.shape[0]  # head_dim // 2

    # freq[i] = base ** (-2i / head_dim), so -log(freq[1]) = 2 * log(base) / head_dim
    log_step = -jnp.log(freq[1])

    def correction_dim(num_rotations: float) -> Array:
        # (fractional) index of the frequency that does num_rotations turns over the original context
        return jnp.log(YARN_ORIGINAL_MAX_LEN / (num_rotations * 2 * jnp.pi)) / log_step

    low = jnp.maximum(jnp.floor(correction_dim(YARN_BETA_FAST)), 0)
    high = jnp.minimum(jnp.ceil(correction_dim(YARN_BETA_SLOW)), 2 * n - 1)

    # ramp = 0 keeps the frequency (high freq dims), ramp = 1 divides it by factor (low freq dims)
    ramp = jnp.clip((jnp.arange(n, dtype=jnp.float32) - low) / (high - low), 0.0, 1.0)
    return (freq / YARN_FACTOR) * ramp + freq * (1 - ramp)


class Olmo3(Transformer):
    qk_norm: bool = True
    qk_norm_per_head: bool = False
    post_norm: bool = True
    rms_eps: float = 1e-6
    rope_correction: RopeCorrection = yarn_rope_correction
    rope_scale: float = YARN_ATTENTION_FACTOR

    def __post_init__(self):
        # sliding window attention isn't implemented; up to SLIDING_WINDOW tokens it is identical to full attention
        # TODO: fix this
        if self.sequence_len > SLIDING_WINDOW:
            raise ValueError(
                f"Olmo3 sequence_len must be <= {SLIDING_WINDOW} (sliding window attention is not implemented), "
                f"got {self.sequence_len}"
            )
        super().__post_init__()

    def uses_rope_correction(self, layer: int) -> bool:
        # yarn only applies to the full attention layers; sliding window layers use plain rope (rope_theta only)
        return (layer + 1) % FULL_ATTENTION_EVERY == 0

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
            # attention norms
            r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": r"Block_\1/GroupedQueryAttention_0/RMSNorm_0.gamma",
            r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": r"Block_\1/GroupedQueryAttention_0/RMSNorm_1.gamma",
            # mlp
            r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": r"Block_\1/FeedForward_0/Dense_0.kernel",
            r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": r"Block_\1/FeedForward_0/Dense_1.kernel",
            r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": r"Block_\1/FeedForward_0/Dense_2.kernel",
            # block norms (post-norm: RMSNorm_0 follows attention, RMSNorm_1 follows the mlp)
            r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": r"Block_\1/RMSNorm_0.gamma",
            r"model\.layers\.([0-9]+)\.post_feedforward_layernorm\.weight": r"Block_\1/RMSNorm_1.gamma",
            # final rms
            r"model\.norm\.weight": "RMSNorm_0.gamma",
            r"lm_head\.weight": "Dense_0.kernel",
        }

    @property
    def reverse_hf_mapping(self):
        return {
            r"token_emb\.embedding": r"model.embed_tokens.weight",
            # block norms
            r"Block_([0-9]+)/RMSNorm_0\.gamma": r"model.layers.\1.post_attention_layernorm.weight",
            r"Block_([0-9]+)/RMSNorm_1\.gamma": r"model.layers.\1.post_feedforward_layernorm.weight",
            # attention projections
            r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_0\.kernel": r"model.layers.\1.self_attn.q_proj.weight",
            r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_1\.kernel": r"model.layers.\1.self_attn.k_proj.weight",
            r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_2\.kernel": r"model.layers.\1.self_attn.v_proj.weight",
            r"Block_([0-9]+)/GroupedQueryAttention_0/Dense_3\.kernel": r"model.layers.\1.self_attn.o_proj.weight",
            # attention norms
            r"Block_([0-9]+)/GroupedQueryAttention_0/RMSNorm_0\.gamma": r"model.layers.\1.self_attn.q_norm.weight",
            r"Block_([0-9]+)/GroupedQueryAttention_0/RMSNorm_1\.gamma": r"model.layers.\1.self_attn.k_norm.weight",
            # mlp
            r"Block_([0-9]+)/FeedForward_0/Dense_0\.kernel": r"model.layers.\1.mlp.gate_proj.weight",
            r"Block_([0-9]+)/FeedForward_0/Dense_1\.kernel": r"model.layers.\1.mlp.up_proj.weight",
            r"Block_([0-9]+)/FeedForward_0/Dense_2\.kernel": r"model.layers.\1.mlp.down_proj.weight",
            # final rms
            r"RMSNorm_0\.gamma": r"model.norm.weight",
            r"Dense_0\.kernel": r"lm_head.weight",
        }
