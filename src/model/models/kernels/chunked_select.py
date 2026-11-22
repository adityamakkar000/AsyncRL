import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
from stax.sharding import AXIS_NAMES

from src.model.utils import act_constrain


def make_chunks(B: int, chunk_size: int):
    assert B % chunk_size == 0, "B must be divisible by chunk_size"
    num_chunks = B // chunk_size
    return num_chunks


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def fused_linear_selection(h, W, targets, chunk_size=1024) -> Array:
    return _fwd(h, W, targets, chunk_size)[0]


def _fwd(h, W, targets, chunk_size):
    B, D = h.shape
    _, _V = W.shape
    chunk_size = min(chunk_size, B)

    num_chunks = make_chunks(B, chunk_size)
    chunked_hidden_inputs = act_constrain(
        h.reshape(num_chunks, chunk_size, D), spec=P(None, AXIS_NAMES.data_mesh(), None)
    )  # num_chunks, chunk_size, D
    chunked_targets = act_constrain(
        targets.reshape(num_chunks, chunk_size), spec=P(None, AXIS_NAMES.data_mesh())
    )  # num_chunks, chunk_size

    W_gather = act_constrain(W, spec=P())

    chunked_hidden_inputs, W_gather = jax.tree.map(lambda x: x.astype(jnp.float32), (chunked_hidden_inputs, W_gather))

    def body_fn(carry, chunk):
        hidden_chunk, target_chunk = chunk
        logits_chunked = hidden_chunk @ W_gather  # chunk_size x V
        log_softmax = jax.nn.log_softmax(logits_chunked, axis=-1)  # chunk_size x V
        target_log_probs = jnp.take_along_axis(log_softmax, target_chunk[:, None], axis=-1)[:, 0]  # chunk_size

        return carry, target_log_probs

    _, chunked_output = jax.lax.scan(body_fn, None, (chunked_hidden_inputs, chunked_targets))  # num_chunks x chunk_size

    output = chunked_output.reshape(B)  # (B,)

    return output, (chunked_hidden_inputs, W_gather, chunked_targets)


def _bwd(chunk_size, residuals, dy):
    chunked_hidden_inputs, W_gather, chunked_targets = residuals
    num_chunks, chunk_size, D = chunked_hidden_inputs.shape
    _, V = W_gather.shape

    chunked_dy = act_constrain(
        dy.astype(jnp.float32).reshape(num_chunks, chunk_size), spec=P(None, AXIS_NAMES.data_mesh())
    )

    def body(dW, chunk):
        hidden_chunk, target_chunk, dy_chunk = chunk
        logits_chunked = hidden_chunk @ W_gather  # chunk_size x V

        p = jax.nn.softmax(logits_chunked, axis=-1)  # chunk_size x V
        onehot = jax.nn.one_hot(target_chunk, V, dtype=p.dtype)
        dz = dy_chunk[:, None] * (onehot - p)  # chunk_size x V

        dh_chunk = dz @ W_gather.T  # chunk_size x D
        dW = dW + hidden_chunk.T @ dz  # D x V
        return dW, dh_chunk

    dW, chunked_dh = jax.lax.scan(
        body,
        jnp.zeros_like(W_gather, dtype=jnp.float32),
        (chunked_hidden_inputs, chunked_targets, chunked_dy),
    )

    dh = chunked_dh.reshape(-1, D)  # (B, D)
    d_targets = jnp.zeros((num_chunks * chunk_size), jax.dtypes.float0)
    return dh, dW, d_targets


fused_linear_selection.defvjp(_fwd, _bwd)
