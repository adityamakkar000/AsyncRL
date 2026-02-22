from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from stax import StepFn

from src.data import RLBatch
from src.model import Model

from .config import LossFunction, RLConfig


def get_ratio(token_logprobs: Array, reference_logprobs: Array, epsilon_low, epsilon_high) -> tuple[Array, Array]:
    log_ratio = token_logprobs - reference_logprobs
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
    weight_ratio = jnp.minimum(ratio, clipped_ratio)
    return weight_ratio, ratio


def grpo_loss(token_logprobs: Array, batch: RLBatch, *, config: RLConfig) -> tuple[Array, PyTree]:
    """
    From  https://arxiv.org/pdf/2412.19437
    """
    B, T = token_logprobs.shape

    weight_ratio, ratio = get_ratio(
        token_logprobs, batch.reference_model_logprobs, config.epsilon_low, config.epsilon_low
    )

    advantages = (batch.rewards - batch.group_mean) / batch.group_std

    masked_loss = weight_ratio * advantages[:, None] * batch.token_mask

    loss = jnp.sum(masked_loss, axis=1) / jnp.sum(batch.token_mask, axis=1)
    loss = jnp.mean(loss)

    aux_metrics = {
        "loss": loss,
        "pi_theta_over_pi_old": jnp.sum(ratio * batch.token_mask) / jnp.sum(batch.token_mask),
    }
    return loss, aux_metrics


def dr_grpo_loss(token_logprobs: Array, batch: RLBatch, *, config: RLConfig) -> tuple[Array, PyTree]:
    """
    From https://arxiv.org/pdf/2503.20783
    """

    B, T = token_logprobs.shape

    weight_ratio, ratio = get_ratio(
        token_logprobs, batch.reference_model_logprobs, config.epsilon_low, config.epsilon_low
    )

    advantages = batch.rewards - batch.group_mean
    masked_loss = weight_ratio * advantages[:, None] * batch.token_mask

    total_seq_len = jnp.sum(batch.token_mask)

    loss = jnp.sum(masked_loss, axis=1).mean() / total_seq_len

    aux_metrics = {
        "loss": loss,
        "pi_theta_over_pi_old": jnp.sum(ratio * batch.token_mask) / jnp.sum(batch.token_mask),
    }
    return loss, aux_metrics


def dapo_loss(token_logprobs: Array, batch: RLBatch, *, config: RLConfig) -> tuple[Array, PyTree]:
    """
    From https://arxiv.org/pdf/2503.14476
    """

    B, T = token_logprobs.shape

    weight_ratio, ratio = get_ratio(
        token_logprobs, batch.reference_model_logprobs, config.epsilon_low, config.epsilon_high
    )

    advantages = batch.rewards - batch.group_mean
    masked_loss = weight_ratio * advantages[:, None] * batch.token_mask

    loss = jnp.sum(masked_loss, axis=1).mean() / T

    aux_metrics = {
        "loss": loss,
        "pi_theta_over_pi_old": jnp.sum(ratio * batch.token_mask) / jnp.sum(batch.token_mask),
    }
    return loss, aux_metrics


# TODO: implment RLOO


def get_loss_fn(RLConfig) -> LossFunction:
    match RLConfig.algorithm:
        case "grpo":
            loss = grpo_loss
        case "dr_grpo":
            loss = dr_grpo_loss
        case "dapo":
            loss = dapo_loss
        case _:
            raise ValueError(f"Unknown loss type: {RLConfig.algorithm}")

    return partial(loss, config=RLConfig)


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
        x_logprobs: Array = jnp.take_along_axis(x_logprobs, batch.tokens[..., None], axis=-1).squeeze(-1)

        loss, aux_metrics = loss_fn(x_logprobs, batch)

        # since we are gradient descenting we want to minimize the loss
        # hence negate the loss you want to maximize
        loss *= -1.0

        return loss, aux_metrics

    return single_step


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
