from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array


def naive_temp_sample(logits: Array, key: Array, *, temperature: float) -> tuple[Array, Array]:
    B, T, V = logits.shape
    logits = logits[:, -1, :] / temperature

    next_tokens = jax.random.categorical(key, logits, axis=-1)[:, None]
    next_logprobbs = jnp.take_along_axis(logits, next_tokens, axis=-1)
    return next_tokens, next_logprobbs


def naive_sample(
    logits: Array, key: Array, *, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None
) -> tuple[Array, Array]:
    """
    Sample the next token from the logits using temperature, top-k, and top-p sampling.
    Args:
        logits (Array): The logits from the model. Shape: [batch_size, vocab_size].
        key (Array): The random key for sampling.
    Returns:
        next_tokens (Array): The sampled next tokens. Shape: [batch_size, 1].
        next_logprobs (Array): The log probabilities of the sampled tokens. Shape: [batch_size, 1].
    """
    B, T, V = logits.shape
    logits = logits[:, -1, :] / temperature

    if top_k:
        logits, base_indices = jax.lax.top_k(logits, top_k)
    else:
        base_indices = jnp.tile(jnp.arange(V), (B, 1))
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    if top_p is not None:
        sort_idx = jnp.argsort(-log_probs, axis=-1)
        sorted_probs = jnp.take_along_axis(log_probs, sort_idx, axis=-1)
        sorted_logits = jnp.take_along_axis(logits, sort_idx, axis=-1)

        mask = jnp.cumsum(sorted_probs, axis=-1) <= top_p
        mask = mask.at[:, 0].set(True)

        filtered_logits = jnp.where(mask, sorted_logits, -jnp.inf)

        logits = jnp.take_along_axis(filtered_logits, jnp.argsort(sort_idx, axis=-1), axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)

    next_idx = jax.random.categorical(key, logits, axis=-1)[:, None]
    next_tokens = jnp.take_along_axis(base_indices, next_idx, axis=-1)
    next_logprobs = jnp.take_along_axis(log_probs, next_idx, axis=-1)

    return next_tokens, next_logprobs


def _maybe_force_eot(
    next_token: Array,  # [B, 1]
    next_log_prob: Array,  # [B, 1]
    end_of_think_mask: Array,  # [B, 1]
    seq_lens: Array,  # [B]
    *,
    reasoning_budget: int,
    token_sequence: list[int],
):
    think_token = token_sequence[-1]
    total_tokens = len(token_sequence)

    end_of_think_mask = end_of_think_mask | (next_token == think_token)

    for t in range(total_tokens):
        # NOTE: only 1 token in the loop can be inserted at most since the equality is differnt
        # for each token
        insert_token = (seq_lens[:, None] + (total_tokens - t)) == reasoning_budget
        interrupt_mask = insert_token & ~end_of_think_mask
        next_token = jnp.where(interrupt_mask, token_sequence[t], next_token)
        next_log_prob = jnp.where(interrupt_mask, 0.0, next_log_prob)

    end_of_think_mask = end_of_think_mask | interrupt_mask

    return next_token, next_log_prob, end_of_think_mask


def _maybe_force_eos(
    next_token: Array,  # [B, 1]
    next_log_prob: Array,  # [B, 1]
    stop_mask: Array,  # [B, 1]
    seq_lens: Array,  # [B]
    *,
    max_seq_len: int,
    eos_token_id: int,
):
    # using seq_len + 1 means the model can respond for max_seq_len and the max_seq_len + 1 token will be <eos>
    length_stop_mask = stop_mask | (seq_lens[:, None] + 1 > max_seq_len)
    eos_stop_mask = next_token == eos_token_id
    stop_mask = eos_stop_mask | length_stop_mask

    next_token = jnp.where(stop_mask, eos_token_id, next_token)
    next_log_prob = jnp.where(length_stop_mask, 0, next_log_prob)

    return next_token, next_log_prob, stop_mask
