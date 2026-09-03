from typing import Optional

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
from stax.sharding.main import AXIS_NAMES_ENUM

from ..config import KVCache
from .flash_attention import SegmentIds, flash_attention
from .norm import RMSNorm
from .rope import apply_rope

DP = AXIS_NAMES_ENUM.DP.value
FSDP = AXIS_NAMES_ENUM.FSDP.value
CP = AXIS_NAMES_ENUM.CP_ULYSSES.value

QKV_SPEC = P((DP, FSDP), CP, None, None)
OUT_SPEC = P((DP, FSDP), None, CP, None)


def flash_attention_naive(q, k, v, mask, sm_scale):
    return flash_attention(q, k, v, sm_scale=sm_scale, segment_ids=SegmentIds(mask, mask), causal=True)


flash_attention_sharded = jax.shard_map(
    flash_attention_naive,
    in_specs=(QKV_SPEC, QKV_SPEC, QKV_SPEC, P(("dp", "fsdp"), None), None),
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

    @property
    def kv_group_size(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def d_out(self) -> int:
        return self.n_heads * self.head_dim

    def gqa(self, q: Array, k: Array, v: Array, mask: Array, kv_cache: KVCache) -> tuple[Array, KVCache]:
        t_start = kv_cache.length
        T = q.shape[1]

        k, v = jax.tree.map(
            lambda cache, val: jax.lax.dynamic_update_slice_in_dim(cache, val.astype(cache.dtype), t_start, axis=1),
            (kv_cache.k, kv_cache.v),
            (k, v),
        )
        kv_cache = KVCache(k=k, v=v, length=t_start + T)

        q = einops.rearrange(q, "b t (g r) d -> b t g r d", g=k.shape[-2])
        wei = jnp.einsum("btgrd, bTgd -> btTgr", q, k) * (self.head_dim**-0.5)
        wei = einops.rearrange(wei, "b t T g r -> b (g r) t T")

        wei = jnp.where(mask, wei, -jnp.inf)
        wei = jax.nn.softmax(wei, axis=-1)
        wei = jnp.where(jnp.isnan(wei), 0.0, wei)

        wei = einops.rearrange(wei, "b (g r) t T -> b t T g r", g=k.shape[2])
        out = jnp.einsum("btTgr, bTgd -> btgrd", wei, v)

        return einops.rearrange(out, "b t g r d -> b t (g r d)"), kv_cache

    def flash_gqa(self, q: Array, k: Array, v: Array, mask: Array) -> Array:
        q = einops.rearrange(q, "... t h d -> ... h t d", d=self.head_dim)
        k = einops.rearrange(k, "... t g d -> ... g t d", d=self.head_dim)
        v = einops.rearrange(v, "... t g d -> ... g t d", d=self.head_dim)

        initializing = self.is_mutable_collection("params")
        if not initializing:
            q, k, v = (jax.lax.with_sharding_constraint(t, QKV_SPEC) for t in (q, k, v))

        k = jnp.repeat(k, self.kv_group_size, axis=1)
        v = jnp.repeat(v, self.kv_group_size, axis=1)

        sm_scale = self.head_dim**-0.5
        fn = flash_attention_naive if initializing else flash_attention_sharded
        out = fn(q, k, v, mask, sm_scale)

        if not initializing:
            out = jax.lax.with_sharding_constraint(out, OUT_SPEC)

        return einops.rearrange(out, "b h t d -> b t (h d)")

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Array,
        rope_matrix: tuple[Array, Array],
        kv_cache: Optional[KVCache] = None,
    ):
        assert self.n_heads % self.n_kv_heads == 0, "Number of heads must be divisible by number of kv heads"
        assert self.model_dim % self.n_heads == 0, "Model dim must be divisible by number of heads"

        q = nn.Dense(features=self.d_out, use_bias=False, dtype=self.activation_dtype)(x)
        k = nn.Dense(features=self.n_kv_heads * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
        v = nn.Dense(features=self.n_kv_heads * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)

        q = einops.rearrange(q, "... t (h d) -> ... t h d", d=self.head_dim)
        k = einops.rearrange(k, "... t (g d) -> ... t g d", d=self.head_dim)
        v = einops.rearrange(v, "... t (g d) -> ... t g d", d=self.head_dim)

        if self.qk_norm:
            q = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(q)
            k = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(k)

        q = apply_rope(q, rope_matrix[0], rope_matrix[1])
        k = apply_rope(k, rope_matrix[0], rope_matrix[1])

        q, k, v = jax.tree.map(lambda t: t.astype(jnp.float32), (q, k, v))

        if kv_cache:
            out, kv_cache = self.gqa(q, k, v, mask, kv_cache)
        else:
            out = self.flash_gqa(q, k, v, mask)

        out = out.astype(self.activation_dtype)
        out = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(out)

        return out, kv_cache
