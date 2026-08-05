import abc
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from src.data import RLBatch
from src.model import Model

from .config import LossConfig


class LossFunction(abc.ABC):
    name: str = "baseLoss"

    def __init__(self, loss_config: LossConfig):
        self.config = loss_config

    @abc.abstractmethod
    def compute_advantage(self, batch: RLBatch) -> Array:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def compute_normalization(self) -> Callable[[int | Array, PyTree], int | Array]:
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_loss(self, x_logprobs: Array, advantages: Array, batch: RLBatch) -> Array:
        """
        Compute the loss from the pre-computed PPO-clipped objective.
        Args:
            x_logprobs (Array): Log probabilities of the current policy. Shape: [B, T].
            advantages (Array): Computed advantages for the batch. Shape: [B].
            batch (RLBatch): The batch of data.
        Returns:
            loss (Array): The computed scalar loss.
        """
        raise NotImplementedError()

    def __call__(self, model: Model, params: PyTree, batch: RLBatch, train: bool = True) -> tuple[Array, PyTree]:
        x_logprobs = model.get_logprobs(params, batch.tokens, batch.seq_lens)  # type: ignore

        batch = batch.replace(  # type: ignore
            tokens=batch.tokens[:, 1:],
            reference_model_logprobs=batch.reference_model_logprobs[:, 1:],
            token_mask=batch.token_mask[:, 1:],
        )

        advantages = self.compute_advantage(batch)
        loss = -1 * self.compute_loss(x_logprobs, advantages, batch)  # negate loss since grad descent

        safe_reference_logprobs = jnp.where(
            jnp.isfinite(batch.reference_model_logprobs), batch.reference_model_logprobs, 0.0
        )
        ratio = jnp.exp(x_logprobs - safe_reference_logprobs)

        # metrics will be reduced by compute_normalization function
        aux_metrics = {
            "loss": loss,
            "is_ratio": jnp.sum(ratio * batch.token_mask),
        }

        return loss, aux_metrics


class CISPOLoss(LossFunction):
    name: str = "CISPO"

    def __init__(self, loss_config: LossConfig, epsilon: float = 4):
        super().__init__(loss_config)
        self.epsilon = epsilon

    def compute_advantage(self, batch: RLBatch) -> Array:
        return batch.rewards - batch.group_mean

    def compute_loss(self, x_logprobs: Array, advantages: Array, batch: RLBatch) -> Array:
        ratio = jnp.exp(x_logprobs - batch.reference_model_logprobs)
        min_ratio = jax.lax.stop_gradient(jnp.minimum(ratio, self.epsilon))
        token_loss = jnp.sum(advantages[:, None] * min_ratio * x_logprobs * batch.token_mask)
        return token_loss

    @property
    def compute_normalization(self) -> Callable[[int | Array, PyTree], int | Array]:
        return lambda denom, batch: denom + jnp.sum(batch.token_mask)


class RLOOLoss(LossFunction):
    name: str = "RLOO"

    def __init__(self, loss_config: LossConfig):
        super().__init__(loss_config)
        assert loss_config.inference_config.group_size > 1, (
            f"RLOO requires group_size > 1 for leave-one-out baseline, got {loss_config.inference_config.group_size}"
        )

    def compute_advantage(self, batch: RLBatch) -> Array:
        G = self.config.inference_config.group_size
        loo_mean = (G * batch.group_mean - batch.rewards) / (G - 1)
        advantages = batch.rewards - loo_mean
        return advantages

    def compute_loss(self, x_logprobs: Array, advantages: Array, batch: RLBatch) -> Array:
        token_loss = x_logprobs * batch.token_mask * advantages[:, None]
        return token_loss.sum()

    @property
    def compute_normalization(self) -> Callable[[int | Array, PyTree], int | Array]:
        return lambda denom, batch: denom + jnp.sum(batch.token_mask)
