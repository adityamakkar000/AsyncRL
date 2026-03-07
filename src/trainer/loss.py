import functools
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from stax import StepFn
from stax import staxLogger as logger

from src.data import RLBatch
from src.model import Model

from .config import LossConfig, LossFunction

ALGO_FN = Callable[[Array, Array, RLBatch, LossConfig], Array]
GLOBAL_DICT: dict[str, ALGO_FN] = {}


def register_algorithim(name: str) -> Callable[[ALGO_FN], ALGO_FN]:
    """Decorator to register an algorithm function. The function should take (x_logprobs, token_mask, batch, config) and return a scalar loss."""

    def decorator(fn: ALGO_FN) -> ALGO_FN:
        GLOBAL_DICT[name] = fn
        return fn

    return decorator


@register_algorithim("cispo")
def cispo_loss(x_logprobs: Array, token_mask: Array, batch: RLBatch, config: LossConfig) -> Array:
    """CISPO loss (https://arxiv.org/pdf/2506.13585)"""

    advantages = batch.rewards - batch.group_mean

    reference_logprobs = batch.reference_model_logprobs[:, 1:]
    reference_logprobs = jnp.where(jnp.isfinite(reference_logprobs), reference_logprobs, 0.0)
    ratio = jnp.exp(x_logprobs - reference_logprobs)
    logger.info("CISPO uses only epsilon-high for clipping ")
    min_ratio = jax.lax.stop_gradient(jnp.minimum(ratio, config.rl_config.epsilon_high))

    token_loss = advantages[:, None] * min_ratio * x_logprobs * token_mask
    total_tokens = jnp.sum(token_mask)
    token_loss = jnp.sum(token_loss) / total_tokens

    return token_loss


@register_algorithim("rloo")
def rloo_loss(x_logprobs: Array, token_mask: Array, batch: RLBatch, config: LossConfig) -> Array:
    """RLOO loss (https://arxiv.org/pdf/2402.14740)."""
    G = config.inference_config.group_size

    loo_mean = (G * batch.group_mean - batch.rewards) / (G - 1)
    advantages = batch.rewards - loo_mean

    token_sum = jnp.sum(x_logprobs * token_mask * advantages[:, None], axis=1)
    seq_mean = token_sum.mean()

    return seq_mean


def get_loss_fn(config: LossConfig) -> LossFunction:
    """Return (loss_fn, normalize_adv_by_std, epsilon_low, epsilon_high) for the algorithm."""
    if config.rl_config.algorithm not in GLOBAL_DICT:
        raise ValueError(f"Got algorithm {config.rl_config.algorithm}, expected one of {list(GLOBAL_DICT.keys())}")

    return functools.partial(GLOBAL_DICT[config.rl_config.algorithm], config=config)


def get_single_step(config: LossConfig) -> StepFn:
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
