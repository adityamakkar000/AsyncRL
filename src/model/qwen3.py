from .transformer import Transformer


class Qwen3(Transformer):
    qk_norm: bool = True
    rms_eps: float = 1e-6

    @property
    def hf_mapping(self):
        return {
            # embedding
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

    @property
    def reverse_hf_mapping(self):
        return {
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
