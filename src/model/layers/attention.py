import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
from stax.sharding.main import AXIS_NAMES_ENUM

from ..config import KVCache
from ..utils import dynamic_update_rows
from .flash_attention import SegmentIds, gqa_flash_attention, gqa_reference
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
    rms_eps: float = 1e-6

    @nn.compact
    def __call__(
        self,
        x: Array,
        masks: tuple[Array, Array],
        rope_matrix: tuple[Array, Array],
        kv_cache: KVCache | None = None,
    ):
        assert self.n_heads % self.n_kv_heads == 0, "Number of heads must be divisible by number of kv heads"
        assert self.model_dim % self.n_heads == 0, "Model dim must be divisible by number of heads"

        q = nn.Dense(features=self.n_heads * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
        k = nn.Dense(features=self.n_kv_heads * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
        v = nn.Dense(features=self.n_kv_heads * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
        q, k, v = jax.tree.map(lambda t: einops.rearrange(t, "b t (h d) -> b t h d", d=self.head_dim), (q, k, v))

        if self.qk_norm:
            q = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(q)
            k = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(k)

        q, k = jax.tree.map(lambda t: apply_rope(t, *rope_matrix), (q, k))
        q, k, v = jax.tree.map(lambda t: t.astype(jnp.float32), (q, k, v))

        train = kv_cache is None
        T = q.shape[1]
        if not train:
            t_start = kv_cache.length
            k, v = jax.tree.map(
                lambda cache, val: dynamic_update_rows(cache, val.astype(cache.dtype), t_start),
                (kv_cache.k, kv_cache.v),
                (k, v),
            )
            kv_cache = KVCache(k=k, v=v, length=t_start + T)

        q, k, v = jax.tree.map(lambda t: einops.rearrange(t, "b t h d -> b h t d"), (q, k, v))
        sm_scale = self.head_dim**-0.5

        if train and not self.is_mutable_collection("params"):
            q, k, v = jax.tree.map(lambda t: jax.lax.with_sharding_constraint(t, QKV_SPEC), (q, k, v))
            out = jax.lax.with_sharding_constraint(flash_attention_sharded(q, k, v, masks, sm_scale), OUT_SPEC)
        elif train or T >= 128:
            out = flash_attention_naive(q, k, v, masks, sm_scale)
        else:
            assert T == 1, "decode only should call gqa"
            out = gqa_reference(q, k, v, None, SegmentIds(*masks), sm_scale=sm_scale)

        out = einops.rearrange(out, "b h t d -> b t (h d)").astype(self.activation_dtype)
        out = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(out)

        return out, kv_cache
