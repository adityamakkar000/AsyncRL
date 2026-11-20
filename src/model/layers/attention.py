import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
from stax.sharding.main import AXIS_NAMES_ENUM

from ..config import KVCache
from .flash_attention import SegmentIds, decode_ragged_attention, gqa_flash_attention
from .norm import RMSNorm
from .rope import apply_rope

DP = AXIS_NAMES_ENUM.DP.value
FSDP = AXIS_NAMES_ENUM.FSDP.value
CP = AXIS_NAMES_ENUM.CP_ULYSSES.value

QKV_SPEC = P((DP, FSDP), CP, None, None)
OUT_SPEC = P((DP, FSDP), None, CP, None)


def flash_attention_naive(q, k, v, masks, sm_scale):
    q_mask, k_mask = masks
    return gqa_flash_attention(q, k, v, sm_scale=sm_scale, segment_ids=SegmentIds(q_mask, k_mask), causal=True)


flash_attention_sharded = jax.shard_map(
    flash_attention_naive,
    in_specs=(QKV_SPEC, QKV_SPEC, QKV_SPEC, P((DP, FSDP), None), None),
    out_specs=QKV_SPEC,
    check_vma=False,
)


class GroupedQueryAttention(nn.Module):
    model_dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    activation_dtype: jnp.dtype = jnp.float32
    qk_norm: bool = False
    qk_norm_per_head: bool = True
    rms_eps: float = 1e-6

    @nn.compact
    def __call__(
        self,
        x: Array,
        seq_lens: Array,
        masks: tuple[Array, Array],
        rope_matrix: tuple[Array, Array],
        kv_cache: KVCache | None = None,
    ):
        assert self.n_heads % self.n_kv_heads == 0, "Number of heads must be divisible by number of kv heads"
        assert self.model_dim % self.n_heads == 0, "Model dim must be divisible by number of heads"

        q = nn.Dense(features=self.n_heads * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
        k = nn.Dense(features=self.n_kv_heads * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
        v = nn.Dense(features=self.n_kv_heads * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)

        if self.qk_norm and not self.qk_norm_per_head:
            # norm over the full projection, before splitting into heads
            q = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(q)
            k = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(k)

        q, k, v = jax.tree.map(lambda t: einops.rearrange(t, "b t (h d) -> b h t d", d=self.head_dim), (q, k, v))

        if self.qk_norm and self.qk_norm_per_head:
            q = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(q)
            k = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(k)

        q, k = jax.tree.map(lambda t: apply_rope(t, *rope_matrix), (q, k))

        T = q.shape[2]
        train = kv_cache is None
        sm_scale = self.head_dim**-0.5

        if train and not self.is_mutable_collection("params"):
            q, k, v = jax.tree.map(lambda t: jax.lax.with_sharding_constraint(t, QKV_SPEC), (q, k, v))
            out = jax.lax.with_sharding_constraint(flash_attention_sharded(q, k, v, masks, sm_scale), OUT_SPEC)
        elif train or T >= 128:
            out = flash_attention_naive(q, k, v, masks, sm_scale)
            if not train:
                k_cache, v_cache = jax.tree.map(
                    lambda cache, val: jax.lax.dynamic_update_slice(cache, val.astype(cache.dtype), (0, 0, 0, 0)),
                    (kv_cache.k, kv_cache.v),
                    (k, v),
                )
                kv_cache = KVCache(k=k_cache, v=v_cache, length=seq_lens)
        else:
            assert T == 1, f"gqa should only be called by decode got {T} shape, full shape{x.shape}"
            out, kv_cache = decode_ragged_attention(q, k, v, kv_cache, sm_scale=sm_scale)

        out = einops.rearrange(out, "b h t d -> b t (h d)").astype(self.activation_dtype)
        out = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(out)

        return out, kv_cache
