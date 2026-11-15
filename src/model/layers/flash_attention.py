# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Flash Attention TPU kernel, modified to support gqa"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from src.model import KVCache

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
VMEM_LIMIT_BYTES = 64 * 1024 * 1024


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

    SegmentIds are used to generate segment mask, which prevents attention between
    different segments in the input sequence. Each array is a list of ids
    (integers).
    Only the token with the same id can attend to each other.

    Attributes:
      q: segment ids along the Q sequence.
      kv: segment ids along the KV sequence.
    """

    q: jax.Array  # [batch_size, q_seq_len]
    kv: jax.Array  # [batch_size, kv_seq_len]


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Tile sizes parameterizing FlashAttention kernels.

    Those parameters have negligible effect on numerics, but affect performance
    greatly.
    """

    block_q: int
    block_k_major: int
    block_k: int
    block_b: int

    block_q_major_dkv: int | None = None
    block_k_major_dkv: int | None = None
    block_k_dkv: int | None = None
    block_q_dkv: int | None = None

    block_k_major_dq: int | None = None
    block_k_dq: int | None = None
    block_q_dq: int | None = None

    def __post_init__(self):
        def verify_major_minor(prefix, suffix, major, minor):
            if minor > major:
                raise ValueError(f"{prefix}{suffix}={minor} should be smaller than {prefix}_major{suffix}={major}")
            if major % minor != 0:
                raise ValueError(f"{prefix}{suffix}={minor} should divide {prefix}_major{suffix}={major}")

        verify_major_minor("block_k", "", self.block_k_major, self.block_k)
        if self.block_q_major_dkv is not None and self.block_q_dkv is not None:
            verify_major_minor("block_q", "_dkv", self.block_q_major_dkv, self.block_q_dkv)
        if self.block_k_major_dkv is not None and self.block_k_dkv is not None:
            verify_major_minor("block_k", "_dkv", self.block_k_major_dkv, self.block_k_dkv)
        if self.block_k_major_dq is not None and self.block_k_dq is not None:
            verify_major_minor("block_k", "_dq", self.block_k_major_dq, self.block_k_dq)

    @property
    def has_backward_blocks(self) -> bool:
        backward_blocks = (
            self.block_q_major_dkv,
            self.block_k_major_dkv,
            self.block_q_dkv,
            self.block_k_dkv,
            self.block_k_major_dq,
            self.block_k_dq,
            self.block_q_dq,
        )
        return all(b is not None for b in backward_blocks)

    @classmethod
    def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model):
        del num_heads, d_model
        q_fit = functools.partial(fit_block, dim=q_seq_len)
        k_fit = functools.partial(fit_block, dim=kv_len)
        return BlockSizes(
            block_q=q_fit(1024),
            block_k_major=k_fit(1024),
            block_k=128,
            block_b=min(2, batch_size),
            block_q_major_dkv=q_fit(2048),
            block_k_major_dkv=k_fit(2048),
            block_k_dkv=128,
            block_q_dkv=128,
            block_k_major_dq=k_fit(1024),
            block_k_dq=128,
            block_q_dq=q_fit(2048),
        )


def fit_block(block: int, dim: int) -> int:
    while block > NUM_LANES and (block > dim or dim % block != 0):
        block //= 2
    return block


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "sm_scale",
        "block_sizes",
        "debug",
    ],
)
def gqa_flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_kv_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_kv_heads, kv_seq_len, d_model]
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    segment_ids=None,  # q of [batch_size, q_seq_len] and kv of [batch_size, kv_seq_len]
    *,
    causal: bool = False,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes | None = None,
    debug: bool = False,
):
    batch_size, num_heads, q_seq_len, d_model = q.shape
    batch_size_k, num_heads_k, kv_seq_len, d_model_k = k.shape
    batch_size_v, num_heads_v, kv_seq_len_v, d_model_v = v.shape
    if batch_size != batch_size_k or batch_size != batch_size_v:
        raise ValueError(
            f"Batch size mismatch: got {batch_size}, {batch_size_k} and {batch_size_v} (for q, k, v respectively)"
        )
    if num_heads_v != num_heads_k:
        raise ValueError(
            f"Head count mismatch: got {num_heads}, {num_heads_k}, {num_heads_v} (for q, k, v respectively)"
        )
    if num_heads % num_heads_k != 0:
        raise ValueError(
            f"num kv heads should dvice num heads: got {num_heads_k} and {num_heads} (for kv_heads and total heads respectively)"
        )
    if d_model != d_model_k:
        raise ValueError(f"Model dimension mismatch: got {d_model} and {d_model_k} (for q and k respectively)")
    if d_model != d_model_v:
        raise NotImplementedError("V model dimension unequal to KV model dimension unsupported")
    if kv_seq_len != kv_seq_len_v:
        raise ValueError(f"KV sequence length mismatch: got {kv_seq_len} and {kv_seq_len_v}")
    if ab is not None and ab.shape != (batch_size, num_heads, q_seq_len, kv_seq_len):
        raise ValueError(
            f"Attention bias shape mismatch: expected ({batch_size=},"
            f" {num_heads=}, {q_seq_len=}, {kv_seq_len=}), got {ab.shape}"
        )
    if segment_ids is not None:
        if segment_ids.q.shape != (batch_size, q_seq_len):
            raise ValueError(
                f"Q segment ids shape mismatch: expected ({batch_size=}, {q_seq_len=},), got {segment_ids.q.shape}"
            )
        if segment_ids.kv.shape != (batch_size, kv_seq_len):
            raise ValueError(
                f"KV segment ids shape mismatch: expected ({batch_size=}, {kv_seq_len=},), got {segment_ids.kv.shape}"
            )
    if block_sizes is None:
        block_sizes = BlockSizes.get_default(batch_size, num_heads, q_seq_len, kv_seq_len, d_model)
    return _gqa_flash_attention(q, k, v, ab, segment_ids, False, causal, sm_scale, block_sizes, debug)


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 10))
def _gqa_flash_attention(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
    return _gqa_flash_attention_impl(
        q,
        k,
        v,
        ab,
        segment_ids,
        save_residuals,
        causal,
        sm_scale,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
        debug,
    )


def _gqa_flash_attention_fwd(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")
    o, l, m = _gqa_flash_attention(q, k, v, ab, segment_ids, True, causal, sm_scale, block_sizes, debug)
    return o, (q, k, v, ab, segment_ids, o, l, m)


def _gqa_flash_attention_bwd(
    save_residuals: bool,
    causal: bool,
    sm_scale: float,
    block_sizes: BlockSizes,
    debug: bool,
    residuals,
    do,
):
    """VJP rule for FlashAttention."""
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")
    (q, k, v, ab, segment_ids, o, l, m) = residuals
    if not block_sizes.has_backward_blocks:
        raise ValueError("Program is being differentiated, but not all backward blocks are specified")

    di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)  # [batch_size, num_heads, q_seq_len]

    dk, dv = _gqa_flash_attention_bwd_dkv(
        q,
        k,
        v,
        ab,
        segment_ids,
        l,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_major_dkv,
        block_k_major=block_sizes.block_k_major_dkv,
        block_k=block_sizes.block_k_dkv,
        block_q=block_sizes.block_q_dkv,
        sm_scale=sm_scale,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        debug=debug,
    )

    dq, ds = _gqa_flash_attention_bwd_dq(
        q,
        k,
        v,
        ab,
        segment_ids,
        l,
        m,
        do,
        di,
        block_q_major=block_sizes.block_q_dq,
        block_k_major=block_sizes.block_k_major_dq,
        block_k=block_sizes.block_k_dq,
        sm_scale=sm_scale,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        debug=debug,
    )
    return dq, dk, dv, ds, None


_gqa_flash_attention.defvjp(fwd=_gqa_flash_attention_fwd, bwd=_gqa_flash_attention_bwd)


MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
    # A block is considered below or on diagonal as long as the bottom left
    # corner of the block is below or on diagonal.
    return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)


def _flash_attention_kernel(q_tile_ref, *args, **kwargs):
    block_b = q_tile_ref.shape[0]
    # If we're not going to tile the softmax, then we can avoid a bunch of VPU ops.
    if kwargs["block_k"] == kwargs["kv_seq_len"]:
        kernel = _flash_attention_kernel_single_batch_single_step
    else:
        kernel = _flash_attention_kernel_single_batch
    for batch_idx in range(block_b):
        kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]
    head_dim = q_tile_ref.shape[-1]

    kv_seq_idx = pl.program_id(3)

    @pl.when(kv_seq_idx == 0)
    def start_new_sequence():
        m_scratch_ref[batch_idx] = jnp.full(m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
        l_scratch_ref[batch_idx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
        acc_scratch_ref[batch_idx] = jnp.zeros(acc_scratch_ref.shape[2:], jnp.float32)

    q_seq_idx = pl.program_id(2)
    if causal:
        should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k_major)
    else:
        should_run = True

    @pl.when(should_run)
    def run():
        @pl.loop(0, block_k_major, step=block_k, unroll=True)
        def _body(start_k):
            m_prev = m_scratch_ref[batch_idx]
            l_prev = l_scratch_ref[batch_idx]
            q = q_tile_ref[batch_idx]  # [block_q, head_dim]
            k = k_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]  # [block_k, head_dim]

            s = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)  # [block_q, block_k]

            # Add attention bias if needed.
            # TODO(tanburn) Should the attention bias be added before or after
            # multiplication by sm_scale?
            if ab_tile_ref is not None:
                ab = ab_tile_ref[(*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))].astype(jnp.float32)
                s += ab

            if sm_scale != 1.0:
                s *= sm_scale

            mask = None
            if q_segment_ids_tile_ref is not None:
                repeats, rem = divmod(block_k, NUM_LANES)
                if rem:
                    raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
                q_segment_ids = jnp.tile(q_segment_ids_tile_ref[batch_idx[0]], (1, repeats))  # [block_q, block_k].
                kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1, pl.dslice(start_k, block_k)]  # [1, block_k].
                mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

            if causal:
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += q_seq_idx * block_q
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += kv_seq_idx * block_k_major + start_k
                causal_mask = col_ids <= row_ids
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

            s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

            m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
            m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].

            block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
            if rem:
                raise NotImplementedError(f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}")
            p = jnp.exp(s - jnp.tile(m_next, (1, block_k_repeats)))

            alpha = jnp.exp(m_prev - m_next)  # Shape [block_q, 128].

            l_corr = alpha * l_prev

            l_next = jnp.sum(p, axis=1)[:, None] + l_corr  # Shape [block_q, 128]

            head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
            l_broadcast = lambda l: jnp.tile(l, (1, head_dim_repeats))
            if rem:
                if head_dim_repeats == 0:
                    l_broadcast = lambda l: l[:, :head_dim]
                else:
                    raise NotImplementedError(f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger")
            l_scratch_ref[batch_idx] = l_next
            m_scratch_ref[batch_idx] = m_next

            l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
            acc_scratch_ref[batch_idx] *= l_broadcast(l_corr * l_next_inv_safe)
            v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
            o_curr = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
            acc_scratch_ref[batch_idx] += o_curr * l_broadcast(l_next_inv_safe)

    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_output():
        o_tile_ref[batch_idx] = acc_scratch_ref[batch_idx].astype(o_tile_ref.dtype)
        if l_ref is not None:
            l_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_ref.dtype)
        if m_ref is not None:
            m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


def _flash_attention_kernel_single_batch_single_step(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref: Any | None = None,
    m_ref: Any | None = None,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]

    assert kv_seq_len == block_k_major == block_k

    q = q_tile_ref[batch_idx]  # [block_q, head_dim]
    k = k_tile_ref[batch_idx]  # [block_k, head_dim]
    s = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)  # [block_q, block_k]

    if ab_tile_ref is not None:
        s += ab_tile_ref[batch_idx].astype(jnp.float32)
    if sm_scale != 1.0:
        s *= sm_scale

    mask = None
    if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
            raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
        q_segment_ids = q_segment_ids_tile_ref[batch_idx[0]]  # [block_q, NUM_LANES].
        q_segment_ids = jnp.tile(q_segment_ids, (1, repeats))  # [block_q, block_k].
        kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

    if causal:
        q_seq_idx = pl.program_id(2)
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = col_ids <= row_ids
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

    m = jnp.max(s, axis=1)[:, None]
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=1)[:, None]
    p /= l

    if m_ref is not None:
        m_ref[batch_idx] = lax.broadcast_in_dim(m, m_ref.shape[2:], range(2))
    if l_ref is not None:
        l_ref[batch_idx] = lax.broadcast_in_dim(l, l_ref.shape[2:], range(2))

    v = v_tile_ref[batch_idx]
    o_tile_ref[batch_idx] = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32).astype(
        o_tile_ref.dtype
    )


def _bytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
    return math.prod(x.shape) * x.dtype.itemsize


def _fwd_cost_estimate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    ab: jax.Array | None,
    segment_ids: SegmentIds | None,
    *,
    causal: bool,
    sm_scale: jax.Array | None,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
    body_cost = pl.estimate_cost(gqa_reference, q, k, v, ab, segment_ids, causal=causal, sm_scale=sm_scale)
    input_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_inputs_specs))
    output_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_outputs_specs))
    return pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes,
    )


def _gqa_flash_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, num_kv_heads, kv_seq_len, _ = k.shape
    group_size = num_heads // num_kv_heads
    _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
    _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
    _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

    # TODO(apaszke): Tile over heads as well.
    grid = (
        pl.cdiv(batch_size, block_b),
        num_heads,
        pl.cdiv(q_seq_len, block_q),
        kv_seq_len // block_k_major,
    )

    def q_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            # If the kv block is skipped, prefetch the next valid kv block, i.e. the
            # 0th one to be used for the next block_q rows.
            next_kv_index = lax.select(
                below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index // group_size, next_kv_index, 0)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            should_run = below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major)
            # If the ab block is skipped, prefetch the next valid ab block, i.e. the
            # 0th kv to be used for the next block_q rows.
            next_q_index = lax.select(
                should_run,
                q_seq_index,
                lax.select(q_seq_index == (q_seq_len // block_q) - 1, 0, q_seq_index + 1),
            )
            next_kv_index = lax.select(should_run, kv_seq_index, 0)
        else:
            next_q_index = q_seq_index
            next_kv_index = kv_seq_index

        return (batch_index, head_index, next_q_index, next_kv_index)

    def o_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    out_shape = [out_shape]
    out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

    if block_k != kv_seq_len:
        m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
        scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
        scratch_shapes = []

    if save_residuals:
        out_specs = [
            *out_specs,
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        ]
        l = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32)
        m = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32)
        out_shape = (*out_shape, l, m)
    else:
        out_specs = [*out_specs, None, None]
        out_shape = (*out_shape, None, None)

    ab_block_spec = pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map) if ab is not None else None

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
            del head_index
            if causal:
                next_kv_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec((block_b, block_q, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec((block_b, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map)

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        ab_block_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
    ]

    o, *aux = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            )
        ),
        cost_estimate=_fwd_cost_estimate(
            q,
            k,
            v,
            ab,
            segment_ids,
            causal=causal,
            sm_scale=sm_scale,
            kernel_inputs_specs=(q, k, v, ab, q_segment_ids, kv_segment_ids),
            kernel_outputs_specs=out_shape,
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids)
    if save_residuals:
        l, m = (v[..., 0] for v in aux[-2:])
        return (o, l, m)
    else:
        return o


def _gqa_flash_attention_dkv_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dk_tile_ref,
    dv_tile_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    q_seq_len: int,
    block_q: int,
    block_k: int,
    group_size: int,
):
    _, _, block_q_major, _ = q_tile_ref.shape
    _, _, block_k_major, _ = k_tile_ref.shape

    q_seq_index = pl.program_id(axis=3)
    kv_seq_index = pl.program_id(axis=1)
    head_index = pl.program_id(axis=2)
    group_index = head_index % group_size

    @pl.when((group_index == 0) & (q_seq_index == 0))
    def start_new_group():
        dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
        dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

    def q_body(j, _):
        start_q = j * block_q

        def k_body(i, _):
            start_k = i * block_k
            k = k_tile_ref[0, 0, pl.ds(start_k, block_k), :]
            v = v_tile_ref[0, 0, pl.ds(start_k, block_k), :]
            q = q_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, head_dim]
            l = l_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
            m = m_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
            do = do_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
            di = di_tile_ref[0, 0, pl.ds(start_q, block_q), :].astype(jnp.float32)  # [block_q, 128]

            capped_logits = lax.dot_general(
                q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
            )  # [block_q_major, block_k]

            if ab_tile_ref is not None:
                ab = ab_tile_ref[
                    0,
                    0,
                    pl.dslice(j * block_q, block_q),
                    pl.dslice(i * block_k, block_k),
                ].astype(jnp.float32)
                capped_logits += ab

            if sm_scale != 1.0:
                capped_logits *= sm_scale

            mask = None
            if q_segment_ids_tile_ref is not None:
                repeats, rem = divmod(block_k, NUM_LANES)
                if rem:
                    raise NotImplementedError()
                q_segment_ids = q_segment_ids_tile_ref[0, pl.ds(start_q, block_q), :]  # [block_q, NUM_LANES].
                q_segment_ids = jnp.tile(q_segment_ids, (1, repeats))  # [block_q, block_k].
                kv_segment_ids = kv_segment_ids_tile_ref[:, 0, pl.ds(start_k, block_k)]  # [1, block_k].
                mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

            if causal:
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += q_seq_index * block_q_major + start_q
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += kv_seq_index * block_k_major + start_k
                causal_mask = col_ids <= row_ids
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

            capped_logits = capped_logits if mask is None else capped_logits + jnp.where(mask, 0.0, mask_value)

            p = jnp.exp(capped_logits - jnp.tile(m, (1, block_k // MIN_BLOCK_SIZE)))
            p = p * jnp.tile(1 / l, (1, block_k // MIN_BLOCK_SIZE))  # [block_q_major, block_k_major]
            dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
            dv_scratch_ref[pl.ds(start_k, block_k), :] += dv.astype(dv_scratch_ref.dtype)

            # di: [block_q, 128]
            # do: [block_q, head_dim]
            # v: [block_k_major, head_dim]
            dp = lax.dot_general(do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)
            ds = (dp - jnp.tile(di, (1, block_k // MIN_BLOCK_SIZE))) * p

            if sm_scale != 1.0:
                ds = ds * sm_scale

            # ds: [block_q_major, block_k_major]
            # q: [block_q_major, head_dim]
            dk = lax.dot(ds.T.astype(do.dtype), q, preferred_element_type=jnp.float32)
            dk_scratch_ref[pl.ds(start_k, block_k), :] += dk.astype(dk_scratch_ref.dtype)

        lax.fori_loop(0, block_k_major // block_k, k_body, None, unroll=True)

    if causal:
        should_run = below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major)
    else:
        should_run = True

    @pl.when(should_run)
    def run():
        lax.fori_loop(0, block_q_major // block_q, q_body, None, unroll=True)

    @pl.when((group_index == group_size - 1) & (q_seq_index == q_seq_len // block_q_major - 1))
    def end_of_group():
        dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref.dtype)
        dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref.dtype)


def _gqa_flash_attention_bwd_dkv(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_q: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    debug: bool = False,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, num_kv_heads, kv_seq_len, _ = k.shape
    group_size = num_heads // num_kv_heads
    _verify_block("block_q_major_dkv", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_q_dkv", "q_seq_len", block_q, q_seq_len)
    _verify_block("block_k_major_dkv", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dkv", "kv_seq_len", block_k, kv_seq_len)

    # Broadcast out scalar values
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    # Preprocess contraction for bwd pass
    di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

    # kv index needs to be before q index since q index is the contractng
    # dimension.
    grid = (
        batch_size,
        kv_seq_len // block_k_major,
        num_heads,
        q_seq_len // block_q_major,
    )

    def qo_index_map(batch_index, kv_seq_index, head_index, q_seq_index):
        if causal:
            # If the q block is skipped, stay at the 0th q block.
            next_q_index = lax.select(
                below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                q_seq_index,
                0,
            )
        else:
            next_q_index = q_seq_index

        return (batch_index, head_index, next_q_index, 0)

    qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    assert qo_spec.block_shape is not None
    assert q.ndim == len(qo_spec.block_shape)
    do_spec = qo_spec
    assert do.ndim == len(qo_spec.block_shape)

    def kv_index_map(batch_index, kv_seq_index, head_index, _):
        return (batch_index, head_index // group_size, kv_seq_index, 0)

    kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, _, head_index, q_seq_index):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def ab_index_map(batch_index, kv_seq_index, head_index, q_seq_index):
        return (batch_index, head_index, q_seq_index, kv_seq_index)

    dab_spec = pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map) if ab is not None else None

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, kv_seq_index, head_index, q_seq_index):
            del head_index
            if causal:
                next_q_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                    q_seq_index,
                    0,
                )
            else:
                next_q_index = q_seq_index
            return (batch_index, next_q_index, 0)

        def kv_segment_ids_index_map(batch_index, kv_seq_index, head_index, _):
            del head_index
            return (batch_index, 0, kv_seq_index)

        q_segment_ids_spec = pl.BlockSpec((1, block_q_major, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec((1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map)

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        qo_spec,
        kv_spec,
        kv_spec,
        dab_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    out_shapes = [
        jax.ShapeDtypeStruct((batch_size, num_kv_heads, kv_seq_len, head_dim), k.dtype),
        jax.ShapeDtypeStruct((batch_size, num_kv_heads, kv_seq_len, head_dim), v.dtype),
    ]

    def dkv_index_map(batch_index, kv_seq_index, head_index, _):
        return (batch_index, head_index // group_size, kv_seq_index, 0)

    dkv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), dkv_index_map)
    out_specs = [dkv_spec, dkv_spec]
    scratch_shapes = [
        pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
        pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
    ]

    kernel = functools.partial(
        _gqa_flash_attention_dkv_kernel,
        block_q=block_q,  # type: ignore
        block_k=block_k,  # type: ignore
        sm_scale=sm_scale,
        causal=causal,
        mask_value=mask_value,
        q_seq_len=q_seq_len,
        group_size=group_size,
    )
    name_scope = f"flash_mha_bwd_dkv_{block_q_major=}_{block_q=}_{block_k_major=}_{block_k=}"
    with jax.named_scope(name_scope):
        dk, dv = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shapes,
            debug=debug,
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=VMEM_LIMIT_BYTES,
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "arbitrary",
                    "arbitrary",
                )
            ),
        )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)
        assert dk.shape == k.shape
        assert dv.shape == v.shape
    return dk, dv


def _gqa_flash_attention_dq_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dq_tile_ref,
    ds_tile_ref,
    dq_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    kv_seq_len: int,
    block_k: int,
):
    _, _, block_k_major, _ = k_tile_ref.shape
    _, _, block_q_major, _ = q_tile_ref.shape

    kv_seq_index = pl.program_id(axis=3)
    q_seq_index = pl.program_id(axis=2)

    @pl.when(kv_seq_index == 0)
    def start_new_sequence():
        dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)

    def body(i, _):
        k_slice = pl.ds(i * block_k, block_k)
        q = q_tile_ref[0, 0, :, :]
        k = k_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
        v = v_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
        l = l_tile_ref[0, 0, :, :]  # [block_q_major, 128]
        m = m_tile_ref[0, 0, :, :]  # [block_q_major, 128]
        do = do_tile_ref[0, 0, :, :]  # [block_q_major, head_dim]
        di = di_tile_ref[0, 0, :].astype(jnp.float32)  # [block_q_major, 128]

        capped_logits = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

        if ab_tile_ref is not None:
            ab = ab_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)].astype(jnp.float32)
            capped_logits += ab

        if sm_scale != 1.0:
            capped_logits *= sm_scale

        mask = None
        if q_segment_ids_tile_ref is not None:
            repeats, rem = divmod(block_k, NUM_LANES)
            if rem:
                raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
            q_segment_ids = jnp.tile(q_segment_ids_tile_ref[0], (1, repeats))  # [block_q, block_k].
            kv_segment_ids = kv_segment_ids_tile_ref[:, 0, k_slice]  # [1, block_k].
            mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

        if causal:
            mask_shape = (block_q_major, block_k)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += q_seq_index * block_q_major
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += kv_seq_index * block_k_major + i * block_k
            causal_mask = col_ids <= row_ids
            mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        capped_logits = capped_logits if mask is None else capped_logits + jnp.where(mask, 0.0, mask_value)

        p = jnp.exp(capped_logits - jnp.tile(m, (1, block_k // MIN_BLOCK_SIZE)))
        p = p * jnp.tile(1 / l, (1, block_k // MIN_BLOCK_SIZE))  # [block_q_major, block_k]

        # di: [block_q_major, 128]
        # do: [block_q_major, head_dim]
        # v: [block_k_major, head_dim]
        dp = jax.lax.dot_general(
            do,
            v,
            TRANS_B_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )
        ds = (dp - jnp.tile(di, (1, block_k // MIN_BLOCK_SIZE))) * p
        # dp = jnp.dot(do, v.T)
        # ds = (dp - (dp * p).sum(axis=1)[:, None]) * p

        if sm_scale != 1.0:
            ds = ds * sm_scale

        if ds_tile_ref is not None:
            ds_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)] = ds.astype(ds_tile_ref.dtype)

        # dp: [block_q_major, block_k]
        # k: [block_k, head_dim]
        dq_scratch_ref[:, :] += lax.dot(
            ds.astype(k.dtype),
            k,
            preferred_element_type=jnp.float32,
        ).astype(dq_scratch_ref.dtype)

    if causal:
        should_run = below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major)
        should_not_run = lax.select(should_run, False, True)
    else:
        should_run = True
        should_not_run = False

    @pl.when(should_run)
    def run():
        lax.fori_loop(0, block_k_major // block_k, body, None, unroll=True)

    @pl.when(should_not_run)
    def zero_out_ds():
        if ds_tile_ref is not None:
            ds_tile_ref[...] = jnp.zeros_like(ds_tile_ref)

    @pl.when(kv_seq_index == kv_seq_len // block_k_major - 1)
    def end_of_kv_sequence():
        dq_tile_ref[0, 0, :, :] = dq_scratch_ref[...].astype(dq_tile_ref.dtype)
        dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _gqa_flash_attention_bwd_dq(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    debug: bool,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, num_kv_heads, kv_seq_len, _ = k.shape
    group_size = num_heads // num_kv_heads
    _verify_block("block_q_dq", "q_seq_len", block_q_major, q_seq_len)
    _verify_block("block_k_major_dq", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k_dq", "block_k", block_k, kv_seq_len)

    # Broadcast out scalar values
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    # Preprocess contraction for bwd pass
    di = jnp.broadcast_to(di[..., None], (*di.shape, block_k_major))

    grid = (
        batch_size,
        num_heads,
        q_seq_len // block_q_major,
        kv_seq_len // block_k_major,
    )

    def qo_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    do_spec = qo_spec

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            # If the kv block is skipped, prefetch the next valid kv block, i.e. the
            # 0th one to be used for the next block_q rows.
            next_kv_index = lax.select(
                below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index // group_size, next_kv_index, 0)

    kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
    assert di_spec.block_shape is not None
    assert di.ndim == len(di_spec.block_shape)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        return (batch_index, head_index, q_seq_index, kv_seq_index)

    dab_spec = pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map) if ab is not None else None

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
            del head_index
            if causal:
                # If the kv block is skipped, prefetch the next valid kv block, i.e. the
                # 0th one to be used for the next block_q rows.
                next_kv_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q_major, kv_seq_index, block_k_major),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec((1, block_q_major, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec((1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map)

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        qo_spec,
        kv_spec,
        kv_spec,
        dab_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
        lm_spec,
        lm_spec,
        do_spec,
        di_spec,
    ]

    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
    ]
    dq_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    out_specs = [
        dq_spec,
        dab_spec,
    ]
    scratch_shapes = [pltpu.VMEM((block_q_major, head_dim), jnp.float32)]  # type: ignore

    kernel = functools.partial(
        _gqa_flash_attention_dq_kernel,
        sm_scale=sm_scale,
        causal=causal,
        mask_value=mask_value,
        block_k=block_k,  # type: ignore
        kv_seq_len=kv_seq_len,
    )
    name_scope = f"flash_mha_bwd_dq_{block_q_major=}_{block_k_major=}_{block_k=}"
    with jax.named_scope(name_scope):
        dq, ds = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shapes,
            debug=debug,
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=VMEM_LIMIT_BYTES,
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
            ),
        )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)

    # dab is just ds
    return dq, ds


# For autograd testing.
def gqa_reference_no_custom_vjp(
    q, # [b, hq, t, d]
    k, # [b, hkv, t, d]
    v, # [b, hkv, t, d]
    ab: jax.Array | None = None,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
):

    n_kv=k.shape[1]
    q = rearrange(q, "b (h g) q c->b h g q c", h=n_kv)
    logits = jnp.einsum("bhgqc,bhkc->bhgqk", q, k)
    if ab is not None:
        ab = rearrange(ab, "b (h g) q k->b h g q k", h=n_kv)
        logits += ab
    if sm_scale != 1.0:
        logits *= sm_scale

    mask = None
    if segment_ids is not None:
        mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
        mask = mask[:, None, None, :, :] # [b, h, g, q, k]

    if causal:
        q_seq_len = q.shape[-2]
        kv_seq_len = k.shape[-2]
        mask_shape = (q_seq_len, kv_seq_len)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = (col_ids <= row_ids)[None, None, None, :, :] # [b, h, g, q, k]
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

    logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

    m = logits.max(axis=-1)
    unnormalized = jnp.exp(logits - m[..., None])
    l = unnormalized.sum(axis=-1)
    weights = unnormalized / l[..., None]
    out = jnp.einsum("bhgqk,bhkc->bhgqc", weights, v)
    out = rearrange(out, "b h g q c->b (h g) q c", h=n_kv)
    if save_residuals:
        return out, l, m
    return out


@functools.partial(jax.jit, static_argnames=["causal", "mask_value", "sm_scale"])
@jax.default_matmul_precision("bfloat16")
def gqa_reference(
    q, # [b, hq, t, d]
    k, # [b, hkv, t, d]
    v, # [b, hkv, t, d]
    ab,
    segment_ids: SegmentIds | None = None,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale=1.0,
):
    return _gqa_reference(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        sm_scale=sm_scale,
        save_residuals=False,
    )


def check_div(n: int, bs: int):
    if n % bs != 0:
        raise ValueError(f"size of {n} could not be divided by {bs}")

def ragged_decode_kernel(
    # prefetch refs
    kv_lengths: jax.ArrayRef,  #[b,]
    # input ref
    q_ref: jax.ArrayRef,
    k_ref: jax.ArrayRef,
    v_ref: jax.ArrayRef,
    k_cache_ref: jax.ArrayRef,
    v_cache_ref: jax.ArrayRef,
    # output ref
    o_ref: jax.ArrayRef,
    k_cache_hbm: jax.ArrayRef,
    v_cache_hbm: jax.ArrayRef,
    # scratch memory
    o_scratch_ref: jax.ArrayRef,
    running_sum_scratch_ref: jax.ArrayRef,
    max_logit_scratch_ref: jax.ArrayRef,
    k_chunk_ref: jax.ArrayRef,
    v_chunk_ref: jax.ArrayRef,
    sems: jax.ArrayRef,  # DMA sems [read/write, k/v, block_b]
    # params,
    kv_seq_len: int,
    block_b: int,
    block_kv: int,
    align: int,
):

    # q_ref [block_b_size, kv_heads, grousp, head_dim]
    # k_ref [block_b_size, kv_heads, head_dim]
    # v_ref [block_b_size, kv_heads, head_dim]
    # k_cache_ref [block_b_size, kv_heads, kv_seq_len, head_dim]
    # v_cache_ref [block_b_size, kv_heads, kv_seq_len, head_dim]

    min_val = jnp.finfo(q_ref.dtype).min
    b, i = pl.program_id(0), pl.program_id(1)

    seq_start = i * block_kv
    pos = seq_start + jax.lax.broadcasted_iota(jnp.int32, (1, 1, block_kv), 2)
    max_blocks = (kv_seq_len // block_kv - 1)
    is_last_block = (i == max_blocks)


    def create_kv_read_write_dmas(batch_idx, kv_len): 
        
        global_batch_idx = b * block_b + batch_idx
        start = pl.multiple_of((kv_len // align) * align, align)

        for scratch_chunk_ref, hbm_ref, sem_read, sem_write in (
            (k_chunk_ref, k_cache_hbm, sems.at[0, 0, batch_idx], sems.at[1, 0, batch_idx]),
            (v_chunk_ref, v_cache_hbm, sems.at[0, 1, batch_idx], sems.at[1, 1, batch_idx]),
        ):

            vmem_chunk = scratch_chunk_ref.at[batch_idx] # [kv_heads, block_kv, d]
            hbm_chunk = hbm_ref.at[global_batch_idx, :, pl.ds(start, align), :]

            read_dma = pltpu.make_async_copy(
                hbm_chunk, 
                vmem_chunk, 
                sem_read
            )
            write_dma = pltpu.make_async_copy(
                vmem_chunk, 
                hbm_chunk,
                sem_write
            )

            yield read_dma, write_dma

    @pl.when(i == 0)
    def _():
        
        q = q_ref[...].astype(jnp.float32)
        k = k_ref[...].astype(jnp.float32)
        v = v_ref[...]

        scores = jnp.sum(q * k[:, :, None, :], axis=-1)
        max_logit_scratch_ref[...] = jnp.broadcast_to(scores[..., None], max_logit_scratch_ref.shape) # (b, h, 1)
        running_sum_scratch_ref[...] = jnp.ones_like(running_sum_scratch_ref) # one since jnp.exp(scores - max_logit) = 1 so sum is 1
        o_scratch_ref[...] = jnp.broadcast_to(v[..., None, :], o_scratch_ref.shape).astype(o_scratch_ref.dtype)

    def loop_fn(idx, _):


        batch_idx = b * block_b + idx
        kv_len =  kv_lengths[batch_idx]
        should_compute =  seq_start <  kv_len

        dmas = list(create_kv_read_write_dmas(idx, kv_len))

        @pl.when(i == 0)
        def _():

            for read_dma, _write_dma in dmas:
                read_dma.start()

        @pl.when(i == max_blocks // 2)
        def _():

            start_idx = kv_len // align * align
            row = jax.lax.broadcasted_iota(jnp.int32, k_chunk_ref.shape[1:], dimension=1) == (kv_len - start_idx)

            for (read_dma, write_dma), (new_ref, scratch_chunk_ref)  in zip(
                dmas, 
                (
                    (k_ref, k_chunk_ref),
                    (v_ref, v_chunk_ref),
                )
            ):
                read_dma.wait() 

                # TODO: quantize here if needed 
                chunk = scratch_chunk_ref.at[idx]
                new = new_ref[idx][:, None, :].astype(chunk.dtype) # [kv_heads, 1, d]
                chunk[...] = jnp.where(row, new, chunk[...])

                write_dma.start()

        @pl.when(should_compute)
        def _():

            q = q_ref[idx]  # [kv_heads, groups, head_dim]
            k = k_cache_ref[idx].astype(q.dtype)  # [heads, block_kv_size, head_dim]
            v = v_cache_ref[idx].astype(q.dtype)  # [heads, block_kv_size, head_dim]

            scores = jnp.einsum(
                "hgd,htd->hgt", q, k, preferred_element_type=jnp.float32
            )  # [kv_heads, heads, block_kv_size]

            scores = jnp.where(
                pos < kv_len, scores, min_val
            )

            prev_max_scores = max_logit_scratch_ref[idx][..., :1] # [heads, groups, 1]
            max_scores = jnp.maximum(jnp.max(scores, axis=-1, keepdims=True), prev_max_scores)  # [heads, group, 1]

            adjustment_factor = jnp.exp(prev_max_scores - max_scores)
            shifted_scores = jnp.exp(scores - max_scores)

            running_sum_scratch_ref[idx] = adjustment_factor * running_sum_scratch_ref[idx] + jnp.sum(
                shifted_scores, axis=-1, keepdims=True
            )
            o_scratch_ref[idx] = adjustment_factor * o_scratch_ref[idx] + jnp.einsum(
                "hgt,htd->hgd", shifted_scores.astype(v.dtype), v, preferred_element_type=jnp.float32
            )

            max_logit_scratch_ref[idx] = jnp.broadcast_to(max_scores, max_logit_scratch_ref.shape[1:])

        @pl.when(is_last_block)
        def _():

            for _read_dma, write_dma in dmas:
                write_dma.wait()


    # loop over every batch to only compute tiles that matter
    jax.lax.fori_loop(0, block_b, loop_fn, init_val=None)

    @pl.when(is_last_block)
    def _():
        running_sum = running_sum_scratch_ref[..., :1]
        out = jnp.where(running_sum == 0, 0.0, o_scratch_ref[...] / running_sum)
        o_ref[...] = out.astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=["block_kv", "block_b"], donate_argnums=(3,))
def _decode_ragged_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: KVCache,
    sm_scale: float,
    block_kv: int = 1024,
    block_b: int = 1,
):

    b, h, _, d = q.shape
    b, h_kv, _, d = k.shape

    kv_seq_len = kv_cache.max_len

    check_div(kv_seq_len, block_kv)
    check_div(b, block_b)
    check_div(d, NUM_LANES)

    kv_lengths = kv_cache.length.astype(jnp.int32)

    q = rearrange(q * sm_scale, "b (h g) 1 d -> b h g d", h=h_kv)
    k = k[..., 0, :] # (b, h, d)
    v = v[..., 0, :] # (b, h, d)
    align = 8 * (4 // jnp.dtype(kv_cache.k.dtype).itemsize)

    assert block_kv % align == 0, "kv block size should be align tile size"

    def kv_block(b, i, kv_lengths):
        last_end = kv_lengths[b * block_b]
        for j in range(1, block_b):
            last_end = jnp.maximum(last_end, kv_lengths[b*block_b + j])
        return jnp.minimum(i, last_end // block_kv)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(b // block_b, kv_seq_len // block_kv),
        in_specs=[
            pl.BlockSpec((block_b, h_kv, h // h_kv, d), lambda b, i, *_: (b, 0, 0, 0)),  # q
            pl.BlockSpec((block_b, h_kv, d), lambda b, i, *p: (b, 0, 0)),  # k
            pl.BlockSpec((block_b, h_kv, d), lambda b, i, *p: (b, 0, 0)),  # v
            pl.BlockSpec((block_b, h_kv, block_kv, d), lambda b, i, *p: (b, 0, kv_block(b, i, *p), 0)),
            pl.BlockSpec((block_b, h_kv, block_kv, d), lambda b, i, *p: (b, 0, kv_block(b, i, *p), 0)),
        ],
        out_specs=[
            pl.BlockSpec((block_b, h_kv, h // h_kv, d), lambda b, i, *_: (b, 0, 0, 0)),
            pl.BlockSpec(memory_space=pl.MemorySpace.ANY),
            pl.BlockSpec(memory_space=pl.MemorySpace.ANY)
        ],
        scratch_shapes=[
            pltpu.VMEM((block_b, h_kv, h // h_kv, d), dtype=jnp.float32), # output accum
            # for max and runnig sum use NUM_LANES since we have to have that as a min length
            # but we will only use the 0th index
            pltpu.VMEM((block_b, h_kv, h // h_kv, NUM_LANES), dtype=jnp.float32), # running sum
            pltpu.VMEM((block_b, h_kv, h // h_kv, NUM_LANES), dtype=jnp.float32), # max logit
            pltpu.VMEM((block_b, h_kv, align, d), dtype=kv_cache.k.dtype),
            pltpu.VMEM((block_b, h_kv, align, d), dtype=kv_cache.v.dtype),
            pltpu.SemaphoreType.DMA((2, 2, block_b)),
        ],
    )

    attn_fn = functools.partial(
        ragged_decode_kernel, kv_seq_len=kv_seq_len, block_b=block_b, block_kv=block_kv, align=align
    )

    out, k_cache, v_cache = pl.pallas_call(
        attn_fn,
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(kv_cache.k.shape, kv_cache.k.dtype),
            jax.ShapeDtypeStruct(kv_cache.v.shape, kv_cache.v.dtype),
        ],
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        input_output_aliases={4: 1, 5: 2}
    )(kv_lengths, q, k, v, kv_cache.k, kv_cache.v)

    kv_cache = kv_cache.replace( # type: ignore
        k=k_cache, 
        v=v_cache, 
        length=kv_cache.length + 1
    )

    out = rearrange(out, "b h g d -> b (h g) 1 d")

    return out, kv_cache 


def decode_ragged_attention(
    q, # [b, hq, 1, d]
    k, # [b, hkv, 1, d]
    v, # [b, hkv, 1, d]
    kv_cache: KVCache,
    sm_scale=1.0,
):
    assert k.shape == v.shape, "kv shapes should match"
    assert q.shape[1] % k.shape[1] == 0, "kv heads should divide heads"
    assert q.shape[2] == 1, "should get 1 query for decode"

    block_kv = fit_block(1024, kv_cache.max_len)
    return _decode_ragged_attention(q, k, v, kv_cache, sm_scale, block_kv=block_kv)

@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _gqa_reference(
    q, # [b, hq, t, d]
    k, # [b, hkv, t, d]
    v, # [b, hkv, t, d]
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
    return gqa_reference_no_custom_vjp(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        sm_scale=sm_scale,
        save_residuals=save_residuals,
    )


def _gqa_reference_fwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
    if save_residuals:
        raise NotImplementedError
    res = gqa_reference_no_custom_vjp(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        sm_scale=sm_scale,
        save_residuals=True,
    )
    assert isinstance(res, tuple)
    out, l, m = res
    return out, (q, k, v, ab, segment_ids, out, l, m)


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "mask_value",
        "sm_scale",
    ],
)
def gqa_reference_bwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    o,
    l,
    m,
    do,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
):
    if sm_scale != 1.0:
        raise NotImplementedError

    n_kv=k.shape[1]
    q = rearrange(q, "b (h g) q c->b h g q c", h=n_kv)
    logits = jnp.einsum("bhgqc,bhkc->bhgqk", q.astype(jnp.float32), k.astype(jnp.float32))

    o = rearrange(o, "b (h g) q c->b h g q c", h=n_kv)
    do = rearrange(do, "b (h g) q c->b h g q c", h=n_kv)

    if ab is not None:
        ab = rearrange(ab,"b (h g) q k->b h g q k", h=n_kv)
        logits += ab

    mask = None
    if segment_ids is not None:
        mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
        mask = mask[:, None, None, :, :]

    if causal:
        q_seq_len = q.shape[-2]
        kv_seq_len = k.shape[-2]
        mask_shape = (q_seq_len, kv_seq_len)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = (col_ids <= row_ids)[None, None, None, :, :]
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

    logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

    unnormalized = jnp.exp(logits - m[..., None])
    p = unnormalized / l[..., None]
    dv = jnp.einsum("bhgpt,bhgpd->bhtd", p, do.astype(jnp.float32)).astype(v.dtype)

    dp = jnp.einsum("bhgpd,bhtd->bhgpt", do.astype(jnp.float32), v.astype(jnp.float32))

    di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)[
        ..., None
    ]  # [batch_size, num_kv_heads, g, q_seq_len]

    ds = (dp - di) * p
    dk = jnp.einsum("bhgsd,bhgst->bhtd", q.astype(jnp.float32), ds).astype(k.dtype)
    dq = jnp.einsum("bhgst,bhtd->bhgsd", ds, k.astype(jnp.float32)).astype(q.dtype)

    # dab is just ds
    dq = rearrange(dq, "b h g q c->b (h g) q c")
    dab = rearrange(ds, "b h g q k->b (h g) q k") if ab is not None else None
    return dq, dk, dv, dab


def _gqa_reference_bwd(
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
    residuals,
    do,
):
    del save_residuals
    q, k, v, ab, segment_ids, o, l, m = residuals
    dq, dk, dv, dab = gqa_reference_bwd(
        q,
        k,
        v,
        ab,
        segment_ids,
        o,
        l,
        m,
        do,
        causal=causal,
        mask_value=mask_value,
        sm_scale=sm_scale,
    )
    return dq, dk, dv, dab, None


_gqa_reference.defvjp(fwd=_gqa_reference_fwd, bwd=_gqa_reference_bwd)


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
    if block > dim:
        raise ValueError(f"{block_name}={block} should be smaller or equal to {dim_name}={dim}")
    if should_divide and dim % block != 0:
        raise ValueError(f"{dim_name}={dim} should be divisible by {block_name}={block}")
