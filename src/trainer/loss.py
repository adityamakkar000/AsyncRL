import functools
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from stax import StepFn
from stax import staxLogger as logger

from src.data import RLBatch
from src.model import Model

from .config import LossFunction, RLConfig


def compute_clipped_objective(
    token_logprobs: Array,
    reference_logprobs: Array,
    advantages: Array,
    epsilon_low: float,
    epsilon_high: float,
) -> Array:
    """Compute the PPO-clipped surrogate objective.

    Args:
        token_logprobs: Log probabilities of the current policy. Shape: [B, T-1].
        reference_logprobs: Log probabilities of the reference policy. Shape: [B, T].
        advantages: Advantage estimates for each sequence. Shape: [B].
        epsilon_low: Clipping parameter for negative advantages.
        epsilon_high: Clipping parameter for positive advantages.
    Returns:
        clipped_objective: min(ratio * A, clip(ratio) * A). Shape [B, T-1].
    """
    reference_logprobs = reference_logprobs[:, 1:]
    safe_reference_logprobs = jnp.where(jnp.isfinite(reference_logprobs), reference_logprobs, 0.0)
    ratio = jnp.exp(token_logprobs - safe_reference_logprobs)
    clipped_ratio = jnp.clip(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)

    adv = advantages[:, None]
    clipped_objective = jnp.minimum(ratio * adv, clipped_ratio * adv)

    return clipped_objective


ALGO_FN = Callable[[Array, Array, RLBatch, RLConfig], Array]
GLOBAL_DICT: dict[str, ALGO_FN] = {}


def register_algorithim(name: str) -> Callable[[ALGO_FN], ALGO_FN]:
    """Decorator to register an algorithm function. The function should take (x_logprobs, token_mask, batch, config) and return a scalar loss."""

    def decorator(fn: ALGO_FN) -> ALGO_FN:
        GLOBAL_DICT[name] = fn
        return fn

    return decorator


@register_algorithim("grpo")
def grpo_loss(x_logprobs: Array, token_mask: Array, batch: RLBatch, config: RLConfig) -> Array:
    """GRPO loss (https://arxiv.org/pdf/2412.19437)."""
    advantages = (batch.rewards - batch.group_mean) / batch.group_std
    logger.info("GRPO uses only epsilon-low for clipping ")
    clipped_objective = compute_clipped_objective(
        x_logprobs, batch.reference_model_logprobs, advantages, config.epsilon_low, config.epsilon_low
    )
    per_seq_loss = jnp.sum(clipped_objective * token_mask, axis=1) / jnp.sum(token_mask, axis=1)
    return jnp.mean(per_seq_loss)


@register_algorithim("dr_grpo")
def dr_grpo_loss(x_logprobs: Array, token_mask: Array, batch: RLBatch, config: RLConfig) -> Array:
    """DR-GRPO loss (https://arxiv.org/pdf/2503.20783)."""
    advantages = batch.rewards - batch.group_mean
    # dr_grpo uses same clipping for positive and negative advantages, handled upstream by setting epsilon_low = epsilon_high
    logger.info("Dr GRPO uses only epsilon-low for clipping ")
    clipped_objective = compute_clipped_objective(
        x_logprobs, batch.reference_model_logprobs, advantages, config.epsilon_low, config.epsilon_low
    )
    per_seq_loss = jnp.sum(clipped_objective * token_mask, axis=1)
    return jnp.mean(per_seq_loss)


@register_algorithim("dapo")
def dapo_loss(x_logprobs: Array, token_mask: Array, batch: RLBatch, config: RLConfig) -> Array:
    """DAPO loss (https://arxiv.org/pdf/2503.14476)."""
    advantages = (batch.rewards - batch.group_mean) / batch.group_std
    clipped_objective = compute_clipped_objective(
        x_logprobs, batch.reference_model_logprobs, advantages, config.epsilon_low, config.epsilon_high
    )
    per_seq_loss = jnp.sum(clipped_objective * token_mask, axis=1)
    return per_seq_loss.mean() / token_mask.sum()


@register_algorithim("rloo")
def rloo_loss(x_logprobs: Array, token_mask: Array, batch: RLBatch, config: RLConfig) -> Array:
    """RLOO loss (https://arxiv.org/pdf/2402.14740)."""
    # @TODO: implment RLOO
    return jnp.array(0.0)


def get_loss_fn(config: RLConfig) -> LossFunction:
    """Return (loss_fn, normalize_adv_by_std, epsilon_low, epsilon_high) for the algorithm."""
    if config.algorithm not in GLOBAL_DICT:
        raise ValueError(f"Got algorithm {config.algorithm}, expected one of {list(GLOBAL_DICT.keys())}")

    return functools.partial(GLOBAL_DICT[config.algorithm], config=config)


def get_single_step(config: RLConfig) -> StepFn:
    """
    Get the RL step function based on the provided configuration.
    Args:
        config (RLConfig): Configuration for the RL training.
    Returns:
        StepFn: A function to be provided to STAX to perform a single training step.
    """
    loss_fn = get_loss_fn(config)

    def single_step(model: Model, params: PyTree, batch: RLBatch, train: bool = True) -> tuple[Array, PyTree]:
        x_logits, kv_cache = model.apply(
            {"params": params}, x=batch.tokens, sequence_lens=batch.seq_lens, kv_cache=None
        )
        x_logprobs = jax.nn.log_softmax(x_logits)
        x_logprobs: Array = jnp.take_along_axis(x_logprobs[:, :-1, :], batch.tokens[:, 1:, None], axis=-1).squeeze(-1)
        token_mask = batch.reference_model_logprobs[:, 1:] != -jnp.inf

        # negate loss since gradient descent and we want to maximize
        loss = -1 * loss_fn(x_logprobs, token_mask, batch)

        safe_reference_logprobs = jnp.where(
            jnp.isfinite(batch.reference_model_logprobs[:, 1:]), batch.reference_model_logprobs[:, 1:], 0.0
        )
        ratio = jnp.exp(x_logprobs - safe_reference_logprobs)
        aux_metrics = {
            "loss": loss,
            "is_ratio": jnp.sum(ratio * token_mask) / jnp.sum(token_mask),
        }

        return loss, aux_metrics

    return single_step  # type: ignore
