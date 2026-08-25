import time
from collections.abc import Callable
from functools import wraps
from statistics import fmean
from typing import Any

import gcsfs
import jax
import jax.numpy as jnp
import stax
from jax.experimental.transfer import start_transfer_server
from jaxtyping import Array
from stax import staxLogger as logger

INFERENCE_REDUCTIONS: dict[str, Callable[[list[float]], float]] = {
    "decode_tps": sum,
    "decode_sps": sum,
    "decode_steps": sum,
    "decode_steps_subbed": sum,
    "ready_rollouts": sum,
}


def reduce_inference_metric(name: str, values: list[float]) -> float:
    return INFERENCE_REDUCTIONS.get(name, fmean)(values)


def setup(setup_fn: Callable[[Any], None], component: str):
    """
    Takes in a function that setups a component (e.g., dataset, model, optimizer)
    and returns a wrapped version that logs the time taken for setup.

    Args:
        setup_fn: The setup function to be wrapped. It should not return anything.
        component: A string representing the component being set up.
    Returns:
        A wrapped setup function that logs the time taken.
    """

    @wraps(setup_fn)
    def wrapper(*args, **kwargs):
        logger.info(f"Setting up {component}")
        start = time.time()
        setup_fn(*args, **kwargs)
        end = time.time()
        logger.info(f"{component} setup complete in {end - start:.2f} seconds.")

    return wrapper


class Key:
    """
    Helper class to manage JAX random keys.
    """

    def __init__(self, seed: int):
        self.key = jax.random.PRNGKey(seed)

    def __call__(self, num_keys: int = 1):
        """
        Generate one or more JAX random keys.
        Args:
            num_keys: Number of keys to generate.
            split_by_process: Whether to fold in the process index for distributed setups.
        Returns:
            A single JAX random key if num_keys is 1, else a list of keys.
        """

        self.key, subkey = jax.random.split(self.key)
        keys = jax.random.split(subkey, (num_keys,))
        return keys if num_keys > 1 else keys[0]


def write_to_gcs(path: str, data: str):
    """Writes data to a file in Google Cloud Storage."""
    if stax.get_rank() == 0:
        fs = gcsfs.GCSFileSystem()
        with fs.open(path.replace("gs://", ""), "w") as f:
            f.write(data)


def naive_temp_sample(logits: Array, key: Array, *, temperature: float) -> tuple[Array, Array]:
    logits = logits[:, -1, :] / temperature

    next_tokens = jax.random.categorical(key, logits, axis=-1)[:, None]
    next_logprobbs = jnp.take_along_axis(logits, next_tokens, axis=-1)
    return next_tokens, next_logprobbs


def naive_sample(
    logits: Array, key: Array, *, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None
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
    B, _T, V = logits.shape
    logits = logits[:, -1, :] / temperature

    if top_k:
        logits, base_indices = jax.lax.top_k(logits, top_k)
    else:
        base_indices = jnp.tile(jnp.arange(V), (B, 1))

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    if top_p is not None:
        sort_idx = jnp.argsort(-log_probs, axis=-1)
        sorted_probs = jnp.take_along_axis(jnp.exp(log_probs), sort_idx, axis=-1)
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
    interrupt_mask = jnp.zeros_like(end_of_think_mask)

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
    # using seq_len + 1 since current seq len doesn't account for the new generated token mean the next token generationed
    # will be the the max seqn len and hence should be <eos>
    length_stop_mask = stop_mask | (seq_lens[:, None] + 1 >= max_seq_len)
    eos_stop_mask = next_token == eos_token_id
    stop_mask = eos_stop_mask | length_stop_mask

    # always set next token for the stop mask to be eos
    # if len_stop_mask is true then we want to force prob 1 (log = 0) to be eos
    # otherwise use the real gen prob
    next_token = jnp.where(stop_mask, eos_token_id, next_token)
    next_log_prob = jnp.where(length_stop_mask, 0, next_log_prob)

    return next_token, next_log_prob, stop_mask


def setup_transfer_server(local_ip: str, port: int):
    backend_client = jax.devices()[0].client
    server = start_transfer_server(
        backend_client,
        f"{local_ip}:{port}",
        [f"{local_ip}:0"] * jax.device_count(),
    )
    return server
