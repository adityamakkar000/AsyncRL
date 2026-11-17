import abc
from collections.abc import Callable

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
        self.teacher_params: PyTree | None = None

    @abc.abstractmethod
    def compute_advantage(self, batch: RLBatch, teacher_params: PyTree = None) -> Array:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def compute_normalization(self) -> Callable[[int | Array, PyTree], int | Array]:
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_loss(
        self, x_logprobs: Array, advantages: Array, batch: RLBatch, teacher_params: PyTree = None
    ) -> tuple[Array, dict[str, Array]]:
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

    def __call__(
        self, model: Model, params: PyTree, batch: RLBatch, teacher_params=None, train: bool = True
    ) -> tuple[Array, PyTree]:
        x_logprobs = model.get_logprobs(params, batch.tokens, batch.seq_lens)

        batch = batch.replace(  # type: ignore
            tokens=batch.tokens[:, 1:],
            reference_model_logprobs=batch.reference_model_logprobs[:, 1:],
            token_mask=batch.token_mask[:, 1:],
        )

        advantages = jax.lax.stop_gradient(self.compute_advantage(batch))
        loss, aux_metrics = self.compute_loss(x_logprobs, advantages, batch, teacher_params)
        loss *= -1  # gradient descent

        log_ratio = x_logprobs - batch.reference_model_logprobs

        # metrics will be reduced by compute_normalization function
        aux_metrics |= {
            "loss": loss,
            "is_ratio": jnp.sum(jnp.exp(log_ratio) * batch.token_mask),
            "kl": jnp.sum(log_ratio * batch.token_mask),
        }

        return loss, aux_metrics


class CISPOLoss(LossFunction):
    name: str = "CISPO"

    def __init__(self, loss_config: LossConfig, epsilon: float = 4):
        super().__init__(loss_config)
        self.epsilon = epsilon

    def compute_advantage(self, batch: RLBatch, teacher_params: PyTree = None) -> Array:
        return batch.rewards - batch.group_mean

    def compute_loss(
        self, x_logprobs: Array, advantages: Array, batch: RLBatch, teacher_params: PyTree = None
    ) -> tuple[Array, dict[str, Array]]:
        ratio = jnp.exp(x_logprobs - batch.reference_model_logprobs)
        min_ratio = jax.lax.stop_gradient(jnp.minimum(ratio, self.epsilon))
        token_loss = jnp.sum(advantages[:, None] * min_ratio * x_logprobs * batch.token_mask)
        return token_loss, {"clip_frac": jnp.sum((ratio > self.epsilon) * batch.token_mask)}

    @property
    def compute_normalization(self) -> Callable[[int | Array, PyTree], int | Array]:
        return lambda denom, batch: denom + jnp.sum(batch.token_mask)


class DrGRPO(LossFunction):
    name: str = "DrGRPO"

    def __init__(self, loss_config: LossConfig, epsilon_low: float = 0.001, epsilon_high: float = 0.001):
        super().__init__(loss_config)
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high

    def compute_advantage(self, batch: RLBatch, teacher_params: PyTree = None) -> Array:
        return batch.rewards - batch.group_mean

    def compute_loss(
        self, x_logprobs: Array, advantages: Array, batch: RLBatch, teacher_params: PyTree = None
    ) -> tuple[Array, dict[str, Array]]:
        ratio = jnp.exp(x_logprobs - batch.reference_model_logprobs)
        loss = jnp.minimum(
            advantages[:, None] * jnp.clip(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high),
            advantages[:, None] * ratio,
        )

        token_loss = jnp.sum(loss * batch.token_mask)
        clipped = (ratio < 1 - self.epsilon_low) | (ratio > 1 + self.epsilon_high)
        return token_loss, {"clip_frac": jnp.sum(clipped * batch.token_mask)}

    @property
    def compute_normalization(self) -> Callable[[int | Array, PyTree], int | Array]:
        return lambda denom, batch: denom + jnp.sum(batch.token_mask)


class MISCispo(LossFunction):
    name: str = "MISCispo"

    def __init__(self, loss_config: LossConfig, epsilon_low: float = 0.0, epsilon_high: float = 5.0):
        super().__init__(loss_config)
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high

    def compute_advantage(self, batch: RLBatch, teacher_params: PyTree = None) -> Array:
        return batch.rewards - batch.group_mean

    def compute_loss(
        self, x_logprobs: Array, advantages: Array, batch: RLBatch, teacher_params: PyTree = None
    ) -> tuple[Array, dict[str, Array]]:
        ratio = jnp.exp(x_logprobs - batch.reference_model_logprobs)
        weight = jax.lax.stop_gradient(jnp.clip(ratio, self.epsilon_low, self.epsilon_high))
        token_loss = jnp.sum(weight * advantages[:, None] * x_logprobs * batch.token_mask)
        clipped = (ratio < self.epsilon_low) | (ratio > self.epsilon_high)
        return token_loss, {"clip_frac": jnp.sum(clipped * batch.token_mask)}

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

    def compute_advantage(self, batch: RLBatch, teacher_params: PyTree = None) -> Array:
        G = self.config.inference_config.group_size
        loo_mean = (G * batch.group_mean - batch.rewards) / (G - 1)
        advantages = batch.rewards - loo_mean
        return advantages

    def compute_loss(
        self, x_logprobs: Array, advantages: Array, batch: RLBatch, teacher_params: PyTree = None
    ) -> tuple[Array, dict[str, Array]]:
        token_loss = x_logprobs * batch.token_mask * advantages[:, None]
        token_loss = token_loss.sum()
        return token_loss, {}

    @property
    def compute_normalization(self) -> Callable[[int | Array, PyTree], int | Array]:
        return lambda denom, batch: denom + jnp.sum(batch.token_mask)

def compute_gae(rewards: Array, values: Array, token_mask: Array, gamma: float = 0.99, lam: float = 0.95) -> Array:
    """
    Args:
        rewards: [B] scalar reward per sequence.
        values: [B, T] value estimates of the state each token was sampled from.
        token_mask: [B, T] mask of tokens to train on.
    Returns:
        advantages: [B, T]
    """
    token_mask = token_mask.astype(values.dtype)
    values = values * token_mask
    zeros = jnp.zeros_like(token_mask[:, :1])

    next_mask = jnp.concatenate([token_mask[:, 1:], zeros], axis=-1)
    next_values = jnp.concatenate([values[:, 1:], zeros], axis=-1) * next_mask

    termination_mask = token_mask * (1 - next_mask)
    token_rewards = termination_mask * rewards[:, None]

    deltas = token_rewards + gamma * next_values - values  # [B, T]

    def step(carry, x):
        delta_t, nm_t = x
        advantage_t = delta_t + gamma * lam * nm_t * carry
        return advantage_t, advantage_t

    _, advantages = jax.lax.scan(
        step, jnp.zeros_like(rewards, dtype=values.dtype), (deltas.T, next_mask.T), reverse=True
    )
    return advantages.T * token_mask


class PPOLoss(LossFunction):
    name: str = "PPO"

    def __init__(
        self,
        loss_config: LossConfig,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        gamma: float = 1.0,
        lam: float = 0.95,
        vf_coeff: float = 0.5,
    ):
        super().__init__(loss_config)
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.gamma = gamma
        self.lam = lam
        self.vf_coeff = vf_coeff

    def compute_advantage(self, batch: RLBatch, teacher_params: PyTree = None, values: Array | None = None) -> Array:
        assert values is not None, "PPO requires value estimates to compute advantages"
        return compute_gae(batch.rewards, jax.lax.stop_gradient(values), batch.token_mask, self.gamma, self.lam)

    def compute_loss(
        self,
        x_logprobs: Array,
        advantages: Array,
        batch: RLBatch,
        teacher_params: PyTree = None,
        values: Array | None = None,
        returns: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        mask = batch.token_mask
        ratio = jnp.exp(x_logprobs - batch.reference_model_logprobs)
        clipped_ratio = jnp.clip(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)

        policy_objective = jnp.minimum(advantages * clipped_ratio, advantages * ratio)
        value_loss = 0.5 * (values - returns) ** 2

        objective = jnp.sum((policy_objective - self.vf_coeff * value_loss) * mask)
        clipped = (ratio < 1 - self.epsilon_low) | (ratio > 1 + self.epsilon_high)

        return objective, {
            "clip_frac": jnp.sum(clipped * mask),
            "value_loss": jnp.sum(value_loss * mask),
            "value_mean": jnp.sum(values * mask),
            "advantage_mean": jnp.sum(advantages * mask),
        }

    def __call__(
        self, model: Model, params: PyTree, batch: RLBatch, teacher_params=None, train: bool = True
    ) -> tuple[Array, PyTree]:
        x_logprobs, values = model.get_logprobs_and_values(params, batch.tokens, batch.seq_lens)

        batch = batch.replace(  # type: ignore
            tokens=batch.tokens[:, 1:],
            reference_model_logprobs=batch.reference_model_logprobs[:, 1:],
            token_mask=batch.token_mask[:, 1:],
        )

        advantages = jax.lax.stop_gradient(self.compute_advantage(batch, values=values))
        returns = jax.lax.stop_gradient(advantages + values)

        loss, aux_metrics = self.compute_loss(x_logprobs, advantages, batch, teacher_params, values=values, returns=returns)
        loss *= -1  # gradient descent

        log_ratio = x_logprobs - batch.reference_model_logprobs

        # metrics will be reduced by compute_normalization function
        aux_metrics |= {
            "loss": loss,
            "is_ratio": jnp.sum(jnp.exp(log_ratio) * batch.token_mask),
            "kl": jnp.sum(log_ratio * batch.token_mask),
        }

        return loss, aux_metrics

    @property
    def compute_normalization(self) -> Callable[[int | Array, PyTree], int | Array]:
        return lambda denom, batch: denom + jnp.sum(batch.token_mask)
