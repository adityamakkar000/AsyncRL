from typing import Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from stax import StepFn

from src.data import RLBatch
from src.model import Model

from .config import LossFunction, RLConfig


def compute_clipped_objective(
    token_logprobs: Array,
    reference_logprobs: Array,
    advantages: Array,
    epsilon_low: float,
    epsilon_high: float,
) -> tuple[Array, Array]:
    """Compute the PPO-clipped surrogate objective.

    Args:
        token_logprobs: Log probabilities of the current policy. Shape: [B, T].
        reference_logprobs: Log probabilities of the reference policy. Shape: [B, T].
        advantages: Advantage estimates for each sequence. Shape: [B].
        token_mask: Mask indicating valid tokens (1 for real tokens, 0 for padding). Shape: [B, T].
        epsilon_low: Clipping parameter for negative advantages.
        epsilon_high: Clipping parameter for positive advantages.
    Returns:
        clipped_objective: min(ratio * A, clip(ratio) * A). Shape [B, T].
        ratio:             π_θ / π_ref (unclipped) Shape [B, T].

    """
    log_ratio = token_logprobs - reference_logprobs
    ratio = jnp.exp(log_ratio)
    clipped_ratio = jnp.clip(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)

    # advantages is [B], broadcast to [B, T]
    adv = advantages[:, None]
    clipped_objective = jnp.minimum(ratio * adv, clipped_ratio * adv)

    return clipped_objective, ratio


def grpo_loss(clipped_objective: Array, token_mask: Array) -> Array:
    """GRPO loss (https://arxiv.org/pdf/2412.19437).

    Normalization: per-sequence mean of masked tokens, then mean across batch.
    """
    per_seq_loss = jnp.sum(clipped_objective * token_mask, axis=1) / jnp.sum(token_mask, axis=1)
    return jnp.mean(per_seq_loss)


def dr_grpo_loss(clipped_objective: Array, token_mask: Array) -> Array:
    """DR-GRPO loss (https://arxiv.org/pdf/2503.20783).

    Normalization: sum tokens per sequence, then mean across batch.
    No per-sequence length division (removes length bias).
    """
    per_seq_loss = jnp.sum(clipped_objective * token_mask, axis=1)
    return jnp.mean(per_seq_loss)


def dapo_loss(clipped_objective: Array, token_mask: Array) -> Array:
    """DAPO loss (https://arxiv.org/pdf/2503.14476).

    Normalization: sum tokens per sequence, mean across batch, divide by T.
    Uses asymmetric clipping (handled upstream via epsilon_low != epsilon_high).
    """
    total_tokens = token_mask.sum()
    per_seq_loss = jnp.sum(clipped_objective * token_mask, axis=1)
    return per_seq_loss.mean() / total_tokens


# Maps algorithm name to (loss_fn, normalize_advantage_by_std, use_asymmetric_clip)
_ALGORITHM_REGISTRY = {
    "grpo": (grpo_loss, True, False),
    "dr_grpo": (dr_grpo_loss, False, False),
    "dapo": (dapo_loss, False, True),
}


def get_loss_fn(config: RLConfig) -> tuple[LossFunction, bool, float, float]:
    """Return (loss_fn, normalize_adv_by_std, epsilon_low, epsilon_high) for the algorithm."""
    if config.algorithm not in _ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    loss_fn, use_std, use_asymmetric = _ALGORITHM_REGISTRY[config.algorithm]

    epsilon_low = config.epsilon_low
    # Symmetric clipping for GRPO/DR-GRPO; asymmetric for DAPO
    epsilon_high = config.epsilon_high if use_asymmetric else config.epsilon_low

    return loss_fn, use_std, epsilon_low, epsilon_high


def get_single_step(config: RLConfig) -> StepFn:
    """
    Get the RL step function based on the provided configuration.
    Args:
        config (RLConfig): Configuration for the RL training.
    Returns:
        StepFn: A function to be provided to STAX to perform a single training step.
    """
    loss_fn, use_std, epsilon_low, epsilon_high = get_loss_fn(config)

    def single_step(model: Model, params: PyTree, batch: RLBatch, train: bool = True) -> tuple[Array, PyTree]:
        x_logits, kv_cache = model.apply(
            {"params": params}, x=batch.tokens, sequence_lens=batch.seq_lens, kv_cache=None
        )
        x_logprobs = jax.nn.log_softmax(x_logits)
        x_logprobs: Array = jnp.take_along_axis(x_logprobs, batch.tokens[..., None], axis=-1).squeeze(-1)

        advantages = batch.rewards - batch.group_mean
        if use_std:
            advantages = advantages / batch.group_std

        clipped_objective, ratio = compute_clipped_objective(
            x_logprobs, batch.reference_model_logprobs, advantages, epsilon_low, epsilon_high
        )

        loss = loss_fn(clipped_objective, batch.token_mask)

        loss = -loss

        aux_metrics = {
            "loss": loss,
            "pi_theta_over_pi_old": jnp.sum(ratio * batch.token_mask) / jnp.sum(batch.token_mask),
        }

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
