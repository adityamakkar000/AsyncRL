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
    def decorator(fn: ALGO_FN) -> ALGO_FN:
        GLOBAL_DICT[name] = fn
        return fn

    return decorator


@register_algorithim("cispo")
def cispo_loss(x_logprobs: Array, token_mask: Array, batch: RLBatch, config: LossConfig) -> Array:
    """CISPO loss (https://arxiv.org/pdf/2506.13585)"""

    advantages = batch.rewards - batch.group_mean

    reference_logprobs = batch.reference_model_logprobs
    reference_logprobs = jnp.where(jnp.isfinite(reference_logprobs), reference_logprobs, 0.0)
    ratio = jnp.exp(x_logprobs - reference_logprobs)
    logger.info("CISPO uses only epsilon-high for clipping ")
    min_ratio = jax.lax.stop_gradient(jnp.minimum(ratio, config.rl_config.epsilon_high))

    token_loss = advantages[:, None] * min_ratio * x_logprobs * token_mask
    total_tokens = jnp.maximum(jnp.sum(token_mask), 1)
    token_loss = jnp.sum(token_loss) / total_tokens

    return token_loss


@register_algorithim("rloo")
def rloo_loss(x_logprobs: Array, token_mask: Array, batch: RLBatch, config: LossConfig) -> Array:
    """RLOO loss (https://arxiv.org/pdf/2402.14740)."""
    G = config.inference_config.group_size

    if G <= 1:
        raise ValueError(f"Group size must be greater than 1 for RLOO, got {G}")

    loo_mean = (G * batch.group_mean - batch.rewards) / (G - 1)
    advantages = batch.rewards - loo_mean

    token_loss = x_logprobs * token_mask * advantages[:, None]

    total_tokens = jnp.sum(token_mask)
    seq_mean = jnp.sum(token_loss) / jnp.maximum(total_tokens, 1)

    return seq_mean


def get_loss_fn(config: LossConfig) -> LossFunction:
    if config.rl_config.algorithm not in GLOBAL_DICT:
        raise ValueError(f"Got algorithm {config.rl_config.algorithm}, expected one of {list(GLOBAL_DICT.keys())}")

    def loss_fn(x_logprobs: Array, token_mask: Array, batch: RLBatch) -> Array:
        return GLOBAL_DICT[config.rl_config.algorithm](x_logprobs, token_mask, batch, config)

    return loss_fn


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
            {"params": params},
            x=batch.tokens,  # type: ignore
            sequence_lens=batch.seq_lens,  # type: ignore
            kv_cache=None,
        )

        batch = batch.replace(  # type: ignore
            tokens=batch.tokens[:, 1:],
            reference_model_logprobs=batch.reference_model_logprobs[:, 1:],
        )

        x_logprobs = jax.nn.log_softmax(x_logits[:, :-1, :], axis=-1)
        x_logprobs: Array = jnp.take_along_axis(x_logprobs, batch.tokens[..., None], axis=-1)[..., 0]
        token_mask = batch.reference_model_logprobs != -jnp.inf

        # negate loss since gradient descent and we want to maximize
        loss = -1 * loss_fn(x_logprobs, token_mask, batch)

        safe_reference_logprobs = jnp.where(
            jnp.isfinite(batch.reference_model_logprobs), batch.reference_model_logprobs, 0.0
        )
        ratio = jnp.exp(x_logprobs - safe_reference_logprobs)
        aux_metrics = {
            "loss": loss,
            "is_ratio": jnp.sum(ratio * token_mask) / jnp.maximum(jnp.sum(token_mask), 1),
        }

        return loss, aux_metrics

    return single_step  # type: ignore
