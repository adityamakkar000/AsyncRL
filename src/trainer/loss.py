from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from stax import StepFn

from src.data import RLBatch
from src.model import Model

from .config import LossFunction, RLConfig


def grpo_loss(token_logprobs: Array, batch: RLBatch, *, config: RLConfig) -> tuple[Array, PyTree]:
    """
    From  https://arxiv.org/pdf/2412.19437
    """
    B, T, V = token_logprobs.shape
    G = config.group_size
    group_rewards = batch.rewards.reshape(B // G, G)
    group_mean = group_rewards.mean(axis=1, keepdims=True) * jnp.ones_like(group_rewards)
    group_std = group_rewards.std(axis=1, keepdims=True) * jnp.ones_like(group_rewards) + 1e-8
    advantages = batch.rewards - group_mean.reshape(
        B,
    ) / group_std.reshape(
        B,
    )

    log_ratio = token_logprobs - batch.reference_model_logprobs
    ratio = jnp.exp(log_ratio)

    clipped_ratio = jnp.clip(ratio, 1.0 - config.epsilon_low, 1.0 + config.epsilon_low)

    weight_ratio = jnp.minimum(ratio, clipped_ratio)
    masked_loss = weight_ratio * advantages[:, None] * batch.token_mask

    # TODO: maybe create a mask from lens instead of reqiuring the whole thing
    loss = jnp.sum(masked_loss, axis=1) / jnp.sum(batch.token_mask, axis=1)
    loss = jnp.mean(loss)

    aux_metrics = {
        "loss": loss,
        "pi_theta_over_pi_old": jnp.mean(ratio),
    }
    return loss, aux_metrics


def dr_grpo_loss(token_logprobs: Array, batch: RLBatch, *, config: RLConfig) -> tuple[Array, PyTree]:
    """
    From https://arxiv.org/pdf/2503.20783
    """

    B, T, V = token_logprobs.shape
    G = config.group_size
    group_rewards = batch.rewards.reshape(B // G, G)
    group_mean = group_rewards.mean(axis=1, keepdims=True) * jnp.ones_like(group_rewards)
    advantages = batch.rewards - group_mean.reshape(
        B,
    )

    log_ratio = token_logprobs - batch.reference_model_logprobs
    ratio = jnp.exp(log_ratio)

    clipped_ratio = jnp.clip(ratio, 1.0 - config.epsilon_low, 1.0 + config.epsilon_low)

    weight_ratio = jnp.minimum(ratio, clipped_ratio)
    # TODO: maybe create a mask from lens instead of reqiuring the whole thing
    masked_loss = weight_ratio * advantages[:, None] * batch.token_mask

    loss = jnp.sum(masked_loss, axis=1).mean()

    aux_metrics = {
        "loss": loss,
        "pi_theta_over_pi_old": jnp.mean(ratio),
    }
    return loss, aux_metrics


def dapo_loss(token_logprobs: Array, batch: RLBatch, *, config: RLConfig) -> tuple[Array, PyTree]:
    """
    From https://arxiv.org/pdf/2503.14476
    """

    B, T, V = token_logprobs.shape
    G = config.group_size
    group_rewards = batch.rewards.reshape(B // G, G)
    group_mean = group_rewards.mean(axis=1, keepdims=True) * jnp.ones_like(group_rewards)
    advantages = batch.rewards - group_mean.reshape(
        B,
    )

    log_ratio = token_logprobs - batch.reference_model_logprobs
    ratio = jnp.exp(log_ratio)

    clipped_ratio = jnp.clip(ratio, 1.0 - config.epsilon_low, 1.0 + config.epsilon_high)

    weight_ratio = jnp.minimum(ratio, clipped_ratio)
    # TODO: maybe create a mask from lens instead of reqiuring the whole thing
    masked_loss = weight_ratio * advantages[:, None] * batch.token_mask

    loss = jnp.sum(masked_loss, axis=1).mean() / T

    aux_metrics = {
        "loss": loss,
        "pi_theta_over_pi_old": jnp.mean(ratio),
    }
    return loss, aux_metrics


def get_loss_fn(RLConfig) -> LossFunction:
    match RLConfig.loss_type:
        case "grpo":
            loss = grpo_loss
        case "dr_grpo":
            loss = dr_grpo_loss
        case "dapo":
            loss = dapo_loss
        case _:
            raise ValueError(f"Unknown loss type: {RLConfig.loss_type}")

    return partial(loss, config=RLConfig)


def get_rl_step_fn(config: RLConfig) -> StepFn:
    """
    Get the RL step function based on the provided configuration.
    Args:
        config (RLConfig): Configuration for the RL training.
    Returns:
        StepFn: A function to be provided to STAX to perform a single training step.
    """
    loss_fn = get_loss_fn(config)

    def step_fn(model: Model, params: PyTree, batch: RLBatch, train: bool = True) -> tuple[Array, PyTree]:
        x_logprobs = model.apply(params, x=batch.tokens, sequence_lens=batch.seq_lens, kv_cache=None, train=train)
        loss, aux_metrics = loss_fn(x_logprobs, batch)

        # since we are gradient descenting we want to minimize the loss
        # hence negate the loss you want to maximize
        loss *= -1.0

        return loss, aux_metrics

    # TODO: (anyone) figure out why this is erroring
    return step_fn  # type: ignore


@jax.jit
def compute_aux_metrics(batch: RLBatch) -> Dict[str, jnp.ndarray]:
    return {
        "mean_reward": jnp.mean(batch.rewards),
        "std_reward": jnp.std(batch.rewards),
        "max_reward": jnp.max(batch.rewards),
        "min_reward": jnp.min(batch.rewards),
        "mean_length": jnp.mean(batch.token_mask.sum(axis=1)),
        "median_length": jnp.median(batch.token_mask.sum(axis=1)),
    }
