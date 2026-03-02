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


def top_k_sampling_kernel(logits: Array, key: Array, *, temperature: float, top_k: int) -> tuple[Array, Array]:
    B, T, V = logits.shape
    logits = logits[:, -1, :] / temperature

    next_tokens = jax.random.categorical(key, logits, axis=-1)[:, None]
    next_logprobbs = jnp.take_along_axis(logits, next_tokens, axis=-1)
    return next_tokens, next_logprobbs


def top_p_sampling_kernel(): ...
