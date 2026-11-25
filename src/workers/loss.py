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

        batch = batch.replace(
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
            "entropy": jnp.sum(-x_logprobs * batch.token_mask),
        }

        return loss, aux_metrics


class ValueLossFunction(LossFunction):
    """Shared critic machinery for value-based losses: GAE advantages, value loss, value bound.
    Subclasses only define the policy term via `policy_objective`."""

    name: str = "valueBase"

    def __init__(
        self,
        loss_config: LossConfig,
        gamma: float = 1.0,
        lam_policy: float = 1.0,
        lam_value: float = 1.0,
        la_gae_alpha: float | None = None,
        vf_coeff: float = 0.5,
        bound_values: str = "none",
        r_min: float = 0.0,
        r_max: float = 1.0,
    ):
        super().__init__(loss_config)
        self.gamma = gamma
        self.lam_policy = lam_policy
        self.lam_value = lam_value
        self.la_gae_alpha = la_gae_alpha
        self.vf_coeff = vf_coeff
        self.bound_values = bound_values
        self.r_min = r_min
        self.r_max = r_max

    @abc.abstractmethod
    def policy_objective(
        self, ratio: Array, x_logprobs: Array, advantages: Array, batch: RLBatch
    ) -> tuple[Array, Array]:
        """Returns (per-token policy objective [B, T], per-token clipped/masked indicator [B, T])."""
        raise NotImplementedError()

    def policy_lambda(self, token_mask: Array) -> float | Array:
        if self.la_gae_alpha is None:
            return self.lam_policy
        lengths = jnp.maximum(jnp.sum(token_mask, axis=-1), 1)
        policy_lambdas = 1 - 1.0 / (self.la_gae_alpha * lengths)
        return jnp.clip(policy_lambdas, 0.0, 1.0)

    def transform_values(self, values: Array) -> Array:
        if self.bound_values == "none":
            return values
        elif self.bound_values == "arctan":
            return self.r_min + (self.r_max - self.r_min) * (0.5 + jnp.arctan(values) / jnp.pi)
        elif self.bound_values == "tanh":
            return self.r_min + (self.r_max - self.r_min) * ((jnp.tanh(values) + 1) / 2)
        elif self.bound_values == "sigmoid":
            return self.r_min + (self.r_max - self.r_min) * jax.nn.sigmoid(values)

    def compute_advantage(self, batch: RLBatch, teacher_params: PyTree = None, values: Array | None = None) -> Array:
        assert values is not None, "value-based losses require value estimates to compute advantages"
        lam = self.policy_lambda(batch.token_mask)
        return compute_gae(batch.rewards, jax.lax.stop_gradient(values), batch.token_mask, self.gamma, lam)

    def compute_returns(self, batch: RLBatch, values: Array) -> Array:
        values = jax.lax.stop_gradient(values)
        return compute_gae(batch.rewards, values, batch.token_mask, self.gamma, self.lam_value) + values

    def value_loss(
        self, model: Model, params: PyTree, batch: RLBatch, teacher_params=None, train: bool = True
    ) -> tuple[Array, PyTree]:
        """Critic-only loss for the extra value updates (critic_updates_per_step - 1 per batch).
        Same bound, target and mask as the value term in __call__; no vf_coeff (only the value head trains here)."""
        values = self.transform_values(model.get_values(params, batch.tokens, batch.seq_lens))
        batch = batch.replace(  # type: ignore
            tokens=batch.tokens[:, 1:],
            reference_model_logprobs=batch.reference_model_logprobs[:, 1:],
            token_mask=batch.token_mask[:, 1:],
        )
        returns = self.compute_returns(batch, values)  # stop-grads values internally
        loss = jnp.sum(0.5 * (values - returns) ** 2 * batch.token_mask)
        return loss, {"value_loss_pre": loss}

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

        policy_objective, clipped = self.policy_objective(ratio, x_logprobs, advantages, batch)
        value_loss = 0.5 * (values - returns) ** 2

        objective = jnp.sum((policy_objective - self.vf_coeff * value_loss) * mask)

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

        values = self.transform_values(values)

        batch = batch.replace(  # type: ignore
            tokens=batch.tokens[:, 1:],
            reference_model_logprobs=batch.reference_model_logprobs[:, 1:],
            token_mask=batch.token_mask[:, 1:],
        )

        advantages = jax.lax.stop_gradient(self.compute_advantage(batch, values=values))
        returns = jax.lax.stop_gradient(self.compute_returns(batch, values))

        loss, aux_metrics = self.compute_loss(
            x_logprobs, advantages, batch, teacher_params, values=values, returns=returns
        )
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
        assert loss_config.group_size > 1, (
            f"RLOO requires group_size > 1 for leave-one-out baseline, got {loss_config.group_size}"
        )

    def compute_advantage(self, batch: RLBatch, teacher_params: PyTree = None) -> Array:
        G = self.config.group_size
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


def compute_gae(
    rewards: Array, values: Array, token_mask: Array, gamma: float = 0.99, lam: float | Array = 0.95
) -> Array:
    """
    Args:
        rewards: [B] scalar reward per sequence.
        values: [B, T] value estimates of the state each token was sampled from.
        token_mask: [B, T] mask of tokens to train on.
        lam: scalar or [B] per-seq lambda (length-adaptive GAE)
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

    lam = jnp.broadcast_to(jnp.asarray(lam, dtype=values.dtype), rewards.shape)

    deltas = token_rewards + gamma * next_values - values  # [B, T]

    def step(carry, x):
        delta_t, nm_t = x
        advantage_t = delta_t + gamma * lam * nm_t * carry
        return advantage_t, advantage_t

    _, advantages = jax.lax.scan(
        step, jnp.zeros_like(rewards, dtype=values.dtype), (deltas.T, next_mask.T), reverse=True
    )
    return advantages.T * token_mask


class PPOLoss(ValueLossFunction):
    """Standard PPO: clip the importance ratio to [1 - eps_low, 1 + eps_high]."""

    name: str = "PPO"

    def __init__(self, loss_config: LossConfig, epsilon_low: float = 0.2, epsilon_high: float = 0.2, **value_kwargs):
        super().__init__(loss_config, **value_kwargs)
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high

    def policy_objective(
        self, ratio: Array, x_logprobs: Array, advantages: Array, batch: RLBatch
    ) -> tuple[Array, Array]:
        clipped_ratio = jnp.clip(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        objective = jnp.minimum(advantages * clipped_ratio, advantages * ratio)
        clipped = (ratio < 1 - self.epsilon_low) | (ratio > 1 + self.epsilon_high)
        return objective, clipped


class DPPOLoss(PPOLoss):
    """DPPO: instead of clipping the ratio, mask tokens whose probability (not ratio)
    moved more than delta in the advantage's direction."""

    name: str = "DPPO"

    def __init__(self, loss_config: LossConfig, delta: float = 0.2, **value_kwargs):
        super().__init__(loss_config, **value_kwargs)
        self.delta = delta

    def policy_objective(
        self, ratio: Array, x_logprobs: Array, advantages: Array, batch: RLBatch
    ) -> tuple[Array, Array]:
        prob_diff = jnp.exp(x_logprobs) - jnp.exp(batch.reference_model_logprobs)
        clipped = jax.lax.stop_gradient(
            ((advantages > 0) & (prob_diff > self.delta)) | ((advantages < 0) & (-prob_diff > self.delta))
        )
        return (1 - clipped) * ratio * advantages, clipped
