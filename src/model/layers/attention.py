from functools import partial

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from ..config import AttentionMask, KVCache, RopeMatrix
from ..utils import DP, FSDP, OUT_SPEC, QKV_SPEC, act_constrain
from .flash_attention import SegmentIds, decode_ragged_attention, gqa_flash_attention
from .norm import RMSNorm
from .rope import apply_rope


def flash_attention_naive(q: Array, k: Array, v: Array, masks: AttentionMask, sm_scale: float) -> Array:
    segment_ids = SegmentIds(masks.q_segment_ids, masks.k_segment_ids)
    return gqa_flash_attention(q, k, v, sm_scale=sm_scale, segment_ids=segment_ids, causal=True)


@partial(
    jax.shard_map,
    in_specs=(QKV_SPEC, QKV_SPEC, QKV_SPEC, P((DP, FSDP), None), None),
    out_specs=QKV_SPEC,
    check_vma=False,
)
def flash_attention_sharded(q: Array, k: Array, v: Array, masks: AttentionMask, sm_scale: float) -> Array:
    return flash_attention_naive(q, k, v, masks, sm_scale)


class GroupedQueryAttention(nn.Module):
    model_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    activation_dtype: jnp.dtype = jnp.float32
    qk_norm: bool = False
    rms_eps: float = 1e-6

    @nn.compact
    def __call__(
        self,
        x: Array,
        masks: AttentionMask,
        rope_matrix: RopeMatrix,
        kv_cache: tuple[Array, KVCache] | None = None,
    ):
        assert self.n_heads % self.n_kv_heads == 0, "Number of heads must be divisible by number of kv heads"
        assert self.model_dim % self.n_heads == 0, "Model dim must be divisible by number of heads"

        q = nn.Dense(features=self.n_heads * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
        k = nn.Dense(features=self.n_kv_heads * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
        v = nn.Dense(features=self.n_kv_heads * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
        q, k, v = jax.tree.map(lambda t: einops.rearrange(t, "b t (h d) -> b h t d", d=self.head_dim), (q, k, v))

        if self.qk_norm:
            q = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(q)
            k = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(k)

        q, k = jax.tree.map(lambda t: apply_rope(t, rope_matrix.sin, rope_matrix.cos), (q, k))

        T = q.shape[2]
        train = kv_cache is None
        sm_scale = self.head_dim**-0.5
        out_cache = None

        if train and not self.is_mutable_collection("params"):
            q, k, v = act_constrain((q, k, v), spec=QKV_SPEC)
            out: Array = flash_attention_sharded(q, k, v, masks, sm_scale)
            out = act_constrain(out, spec=OUT_SPEC)
        elif train or T >= 128:
            out = flash_attention_naive(q, k, v, masks, sm_scale)
            if not train:
                layer, cache = kv_cache
                out_cache = cache.prefill(layer, k, v)
        else:
            assert T == 1, f"gqa should only be called by decode got {T} shape, full shape{x.shape}"
            out, out_cache = decode_ragged_attention(q, k, v, kv_cache, sm_scale=sm_scale)

        out = einops.rearrange(out, "b h t d -> b t (h d)").astype(self.activation_dtype)
        out = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(out)

        return out, out_cache
