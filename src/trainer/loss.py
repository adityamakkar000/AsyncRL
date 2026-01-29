import jax.numpy as jnp
from jaxtyping import Array, PyTree

from src.data import RLBatch
from src.model import Model
from typing import Protocol
from stax import get_steps_fn, StepFn


class LossFunction(Protocol):
    """
    A protocol for loss functions used in reinforcement learning.
    """

    def __call__(self, token_logprobs: Array, token_mask: Array, Rewards: Array) -> tuple[Array, PyTree]:
        """
        Compute the loss given token log probabilities, token mask, and rewards.
        Args:
            token_logprobs (Array): Log probabilities of the tokens.
            token_mask (Array): Mask indicating valid tokens.
            Rewards (Array): Rewards associated with the tokens.
        Returns:
            loss (Array): The computed loss.
            aux_metrics (PyTree): Auxiliary metrics for monitoring.
        """
        ...

#TODO: implement
def grpo_loss(token_logprobs: Array, token_mask: Array, Rewards: Array) -> tuple[Array, PyTree]:
    loss = -jnp.sum(token_logprobs * token_mask * Rewards) / jnp.sum(token_mask)
    aux_metrics = {}
    return loss, aux_metrics

#TODO: implement 
def dr_grpo_loss(token_logprobs: Array, token_mask: Array, Rewards: Array) -> tuple[Array, PyTree]:
    loss = -jnp.sum(token_logprobs * token_mask * Rewards) / jnp.sum(token_mask)
    aux_metrics = {}
    return loss, aux_metrics


#TODO: implement
def dapo_loss(token_logprobs: Array, token_mask: Array, Rewards: Array) -> tuple[Array, PyTree]:

    loss = -jnp.sum(token_logprobs * token_mask * Rewards) / jnp.sum(token_mask)
    aux_metrics = {}
    return loss, aux_metrics

def get_loss_fn(RLConfig) -> LossFunction:
    match RLConfig.loss_type:
        case "grpo":
            return grpo_loss
        case "dr_grpo":
            return dr_grpo_loss
        case "dapo":
            return dapo_loss
        case _:
            raise ValueError(f"Unknown loss type: {RLConfig.loss_type}")

def get_rl_step_fn(RLConfig) -> StepFn: 
    loss_fn = get_loss_fn(RLConfig)
    def step_fn(model : Model , params : PyTree, batch: RLBatch, train:bool=True):
        x_logprobs = model.apply(
            params,
            x=batch.tokens,
            sequence_lens=batch.seq_lens,
            kv_cache=None,
            train=train 
        )
        loss, aux_metrics = loss_fn(x_logprobs, batch.token_train_lens, batch.rewards)
        aux_metrics  |= {
            # other metrics here 
        }

        return loss, aux_metrics    

    return step_fn
