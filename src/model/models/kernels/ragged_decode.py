"""Ragged decode attention TPU kernel with int8/bf16 KV cache"""

# inspired by https://github.com/jax-ml/jax-llm-examples/blob/main/qwen3/qwen3_jax/ragged_attention.py

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from einops import rearrange
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from src.model.config import Int8Array, KVCache
from src.model.models.kernels.common import NUM_LANES, check_div, fit_block


def ragged_decode_kernel(
    # prefetch refs
    kv_lengths: jax.ArrayRef,  # [b,]
    layer_ref: jax.ArrayRef,  # [1,]
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
    k_scratch_ref: jax.ArrayRef,
    v_scratch_ref: jax.ArrayRef,
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

    is_quantized = isinstance(k_cache_ref, Int8Array)

    min_val = jnp.finfo(q_ref.dtype).min
    b, i = pl.program_id(0), pl.program_id(1)

    seq_start = i * block_kv
    pos = seq_start + jax.lax.broadcasted_iota(jnp.int32, (1, 1, block_kv), 2)
    max_blocks = kv_seq_len // block_kv - 1
    is_last_block = i == max_blocks

    def create_kv_read_write_dmas(batch_idx, kv_len) -> list[tuple[Any, Any]]:
        global_batch_idx = b * block_b + batch_idx
        rows = pl.ds(pl.multiple_of(kv_len // align * align, align), align)
        lanes = pl.ds(pl.multiple_of(kv_len // NUM_LANES * NUM_LANES, NUM_LANES), NUM_LANES)
        ds = Int8Array(rows, lanes) if is_quantized else rows  # type : ignore

        def copy(scratch_ref, hbm_ref, kv, rw):
            def f(sc, hbm, sem, d):
                vmem = sc.at[batch_idx]
                hbm = hbm.at[layer_ref[0], global_batch_idx, :, d]
                src, dst = (hbm, vmem) if rw == 0 else (vmem, hbm)
                return pltpu.make_async_copy(src, dst, sem.at[kv, rw, batch_idx])

            return jax.tree.map(f, scratch_ref, hbm_ref, sems, ds)

        dmas = []
        for kv, (scratch_ref, hbm_ref) in enumerate(zip((k_scratch_ref, v_scratch_ref), (k_cache_hbm, v_cache_hbm))):
            read_dma = copy(scratch_ref, hbm_ref, kv, 0)
            write_dma = copy(scratch_ref, hbm_ref, kv, 1)
            dmas.append((read_dma, write_dma))
        return dmas

    @pl.when(i == 0)
    def _():
        q = q_ref[...].astype(jnp.float32)
        k = k_ref[...].astype(jnp.float32)
        v = v_ref[...]

        scores = jnp.sum(q * k[:, :, None, :], axis=-1)
        max_logit_scratch_ref[...] = jnp.broadcast_to(scores[..., None], max_logit_scratch_ref.shape)  # (b, h, 1)
        running_sum_scratch_ref[...] = jnp.ones_like(
            running_sum_scratch_ref
        )  # one since jnp.exp(scores - max_logit) = 1 so sum is 1
        o_scratch_ref[...] = jnp.broadcast_to(v[..., None, :], o_scratch_ref.shape).astype(o_scratch_ref.dtype)

    def loop_fn(idx, _):
        batch_idx = b * block_b + idx
        kv_len = kv_lengths[batch_idx]
        should_compute = seq_start < kv_len

        dmas = create_kv_read_write_dmas(idx, kv_len)

        @pl.when(i == 0)
        def _():
            for read_dma, _write_dma in dmas:
                jax.tree.map(lambda x: x.start(), read_dma)

        @pl.when(i == max_blocks // 2)
        def _():
            align_row = jax.lax.broadcasted_iota(jnp.int32, k_scratch_ref.shape[1:], dimension=1) == (kv_len % align)
            lanes_row = jax.lax.broadcasted_iota(jnp.int32, (k_scratch_ref.shape[1], NUM_LANES), dimension=1) == (
                kv_len % NUM_LANES
            )

            for (read_dma, write_dma), (new_ref, scratch_ref) in zip(
                dmas,
                (
                    (k_ref, k_scratch_ref),
                    (v_ref, v_scratch_ref),
                ),
            ):
                jax.tree.map(lambda x: x.wait(), read_dma)

                new = new_ref[idx][:, None, :]  # [ kv_heads, 1, d]

                def f(mask, new, old):
                    old.at[idx][...] = jnp.where(mask, new.astype(old.dtype), old.at[idx][...])

                if is_quantized:
                    masks, new = Int8Array(align_row, lanes_row), Int8Array.quantize(new)
                else:
                    masks = align_row

                jax.tree.map(f, masks, new, scratch_ref)

                jax.tree.map(lambda x: x.start(), write_dma)

        @pl.when(should_compute)
        def _():
            q = q_ref[idx]  # [kv_heads, groups, head_dim]

            if isinstance(k_cache_ref, Int8Array):
                assert isinstance(v_cache_ref, Int8Array)
                k, k_scale = k_cache_ref.value[idx], k_cache_ref.scale[idx]  # [h, block_kv]
                v, v_scale = v_cache_ref.value[idx], v_cache_ref.scale[idx]  # [h, block_kv]
            else:
                k = k_cache_ref[idx]
                v = v_cache_ref[idx]

            k = k.astype(q.dtype)  # [heads, block_kv_size, head_dim]
            v = v.astype(q.dtype)  # [heads, block_kv_size, head_dim]

            scores = jnp.einsum(
                "hgd,htd->hgt", q, k, preferred_element_type=jnp.float32
            )  # [kv_heads, heads, block_kv_size]

            if is_quantized:
                scores *= k_scale[:, None, :]

            scores = jnp.where(pos < kv_len, scores, min_val)

            prev_max_scores = max_logit_scratch_ref[idx][..., :1]  # [heads, groups, 1]
            max_scores = jnp.maximum(jnp.max(scores, axis=-1, keepdims=True), prev_max_scores)  # [heads, group, 1]

            adjustment_factor = jnp.exp(prev_max_scores - max_scores)
            shifted_scores = jnp.exp(scores - max_scores)

            running_sum_scratch_ref[idx] = adjustment_factor * running_sum_scratch_ref[idx] + jnp.sum(
                shifted_scores, axis=-1, keepdims=True
            )

            if is_quantized:
                shifted_scores *= v_scale[:, None, :]

            o_scratch_ref[idx] = adjustment_factor * o_scratch_ref[idx] + jnp.einsum(
                "hgt,htd->hgd", shifted_scores, v, preferred_element_type=jnp.float32
            )

            max_logit_scratch_ref[idx] = jnp.broadcast_to(max_scores, max_logit_scratch_ref.shape[1:])

        @pl.when(is_last_block)
        def _():
            for _read_dma, write_dma in dmas:
                jax.tree.map(lambda x: x.wait(), write_dma)

    # loop over every batch to only compute tiles that matter
    jax.lax.fori_loop(0, block_b, body_fun=loop_fn, init_val=None)

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
    layer: jax.Array,
    sm_scale: float,
    block_kv: int = 1024,
    block_b: int = 1,
) -> tuple[Array, KVCache]:
    b, h, _, d = q.shape
    b, h_kv, _, d = k.shape

    kv_seq_len = kv_cache.max_len

    check_div(kv_seq_len, block_kv)
    check_div(b, block_b)
    check_div(d, NUM_LANES)

    kv_lengths = kv_cache.length.astype(jnp.int32)

    q = rearrange(q * sm_scale, "b (h g) 1 d -> b h g d", h=h_kv)
    k = k[..., 0, :]  # (b, h, d)
    v = v[..., 0, :]  # (b, h, d)

    align = 8 * (4 // kv_cache.dtype.itemsize)

    assert block_kv % align == 0, "kv block size should be align tile size"

    def kv_block(b, i, kv_lengths):
        last_end = kv_lengths[b * block_b]
        for j in range(1, block_b):
            last_end = jnp.maximum(last_end, kv_lengths[b * block_b + j])
        return jnp.minimum(i, (last_end - 1) // block_kv)

    def kv_tree(value, scale):
        return Int8Array(value, scale) if kv_cache.is_quantized else value

    def kv_cache_index(b, i, kv_lengths, layer):
        return (layer[0], b, 0, kv_block(b, i, kv_lengths), 0)

    def kv_scale_index(b, i, kv_lengths, layer):
        return (layer[0], b, 0, kv_block(b, i, kv_lengths))

    def make_kv_block_spec():
        return [
            kv_tree(
                pl.BlockSpec((None, block_b, h_kv, block_kv, d), kv_cache_index),
                pl.BlockSpec((None, block_b, h_kv, block_kv), kv_scale_index),
            )
        ] * 2

    def make_kv_scratch_space():
        sems = pltpu.SemaphoreType.DMA((2, 2, block_b))
        return [
            kv_tree(
                pltpu.VMEM((block_b, h_kv, align, d), dtype=kv_cache.dtype),
                pltpu.VMEM((block_b, h_kv, NUM_LANES), dtype=jnp.bfloat16),
            )
        ] * 2 + [kv_tree(sems, sems)]

    def make_kv_hbm_spec():
        any_spec = pl.BlockSpec(memory_space=pl.MemorySpace.ANY)
        return [kv_tree(any_spec, any_spec)] * 2

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(b // block_b, kv_seq_len // block_kv),
        in_specs=[
            pl.BlockSpec((block_b, h_kv, h // h_kv, d), lambda b, i, *_: (b, 0, 0, 0)),  # q
            pl.BlockSpec((block_b, h_kv, d), lambda b, i, *_: (b, 0, 0)),  # k
            pl.BlockSpec((block_b, h_kv, d), lambda b, i, *_: (b, 0, 0)),  # v
            *make_kv_block_spec(),
        ],
        out_specs=[
            pl.BlockSpec((block_b, h_kv, h // h_kv, d), lambda b, i, *_: (b, 0, 0, 0)),
            *make_kv_hbm_spec(),
        ],
        scratch_shapes=[
            pltpu.VMEM((block_b, h_kv, h // h_kv, d), dtype=jnp.float32),  # output accum
            # for max and runnig sum use NUM_LANES since we have to have that as a min length
            # but we will only use the 0th index
            pltpu.VMEM((block_b, h_kv, h // h_kv, NUM_LANES), dtype=jnp.float32),  # running sum
            pltpu.VMEM((block_b, h_kv, h // h_kv, NUM_LANES), dtype=jnp.float32),  # max logit
            *make_kv_scratch_space(),
        ],
    )

    attn_fn = functools.partial(
        ragged_decode_kernel, kv_seq_len=kv_seq_len, block_b=block_b, block_kv=block_kv, align=align
    )

    out, k_cache, v_cache = pl.pallas_call(
        attn_fn,
        out_shape=[jax.ShapeDtypeStruct(q.shape, q.dtype), kv_cache.shape_dtype, kv_cache.shape_dtype],
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        input_output_aliases={5: 1, 6: 2, 7: 3, 8: 4} if kv_cache.is_quantized else {5: 1, 6: 2},
    )(kv_lengths, layer[None], q, k, v, kv_cache.k, kv_cache.v)

    out_cache = kv_cache.replace(k=k_cache, v=v_cache)

    out = rearrange(out, "b h g d -> b (h g) 1 d")

    return out, out_cache


def decode_ragged_attention(
    q,  # [b, hq, 1, d]
    k,  # [b, hkv, 1, d]
    v,  # [b, hkv, 1, d]
    kv_cache: tuple[jax.Array, KVCache],
    sm_scale=1.0,
) -> tuple[Array, KVCache]:
    layer, cache = kv_cache
    assert k.shape == v.shape, "kv shapes should match"
    assert q.shape[1] % k.shape[1] == 0, "kv heads should divide heads"
    assert q.shape[2] == 1, "should get 1 query for decode"

    block_kv = fit_block(1024, cache.max_len)
    return _decode_ragged_attention(q, k, v, cache, layer, sm_scale, block_kv=block_kv)
