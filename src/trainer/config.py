from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from jaxtyping import Array
from omegaconf import MISSING

from src.data import DataConfig, RLBatch
from src.inference_engine import InferenceConfig
from src.model import ModelConfig


class LossFunction(Protocol):
    """
    A protocol for loss functions used in reinforcement learning.
    """

    def __call__(self, x_logprobs: Array, token_mask: Array, batch: RLBatch) -> Array:
        """
        Compute the loss from the pre-computed PPO-clipped objective.
        Args:
            x_logprobs (Array): Log probabilities of the current policy. Shape: [B, T].
            token_mask (Array): Mask for valid tokens. Shape: [B, T].
            batch (RLBatch): The batch of data.
        Returns:
            loss (Array): The computed scalar loss.
        """
        ...


@dataclass
class ShardingConfig:
    sharding_type: str = "single"  # "single", "dp", "fsdp"
    opt_state_offload: bool = False
    data_shard_dim: int = 0
    min_bytes_for_fsdp: int = int(1e6)  # 1e6/(1024*1024) = 1MB
    weight_shard_dim: int = 0


@dataclass
class WandBConfig:
    project: str = "Debug"
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


# TODO: actual experiment config
@dataclass
class AnnealedLoss:
    """Config for annealnig the fraction of reasoning trace injected into prompts. Appends to generation prompt."""

    use_annealing: bool = 1
    init_value: float = 1.0  # Fraction of trace at start of annealing (eg 1.0 = full trace)
    end_value: float = 0.0  # Fraction of trace at end of annealing (eg 0.0 = no trace)
    annealing_steps: float = 0.1  # Percentage of total training steps to anneal for (in the beginning)
    schedule: str = "linear"  # "linear", "cosine"


@dataclass
class BestMetric:
    name: str = MISSING  # Name of the metric to monitor
    maximize: bool = False  # Whether to maximize or minimize the metric


@dataclass
class RLConfig:
    algorithm: str = "grpo"  # "grpo", "dr_grpo", "dapo"
    epsilon_high: float = 1.0
    epsilon_low: float = 0.1
    ppo_minibatch_size: int = 32


@dataclass
class LossConfig:
    rl_config: RLConfig = field(default_factory=RLConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    annealing_config: AnnealedLoss = field(default_factory=AnnealedLoss)


@dataclass
class TrainerConfig:
    experiment_name: str = MISSING  # The name of the experiment
    data_config: DataConfig = MISSING  # The configuration for the data module
    model_config: ModelConfig = MISSING  # The configuration for the model
    loss_config: LossConfig = MISSING  # The configuration for the loss function

    sharding_config: ShardingConfig = field(default_factory=ShardingConfig)  # The configuration for sharding

    # training config
    seed: int = 0  # The random seed to use for reproducibility
    # TODO:
    # figure out if we want epochs or steps
    # pros of epochs: more interpretable, cons: lr scheduling, etc
    num_steps: int = 1000  # The number of training epochs
    grad_accum_steps: int = 1  # The number of gradient accumulation steps
    val_interval: int = 100  # The interval (in steps) at which to validate the model
    val_steps: int = 10  # The number of validation steps to run

    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    weight_decay: Optional[float] = None  # The weight decay coefficient
    grad_clip: Optional[float] = None  # The maximum gradient norm for clipping

    # cosine lr
    learning_rate_init: float = 0.0  # The initial learning rate
    learning_rate_peak: float = 1e-4  # The maximum learning rate
    learning_rate_end: float = 1e-5  # The final learning rate
    warmup_steps: float = 0.1  # The fraction of total steps to use for learning rate warmup
    decay_steps: float = 0.9  # The fraction of total steps to use for learning rate decay

    wandb_config: Optional[WandBConfig] = None  # The configuration for Weights & Biases logging
    metrics_to_log: List[str] = field(default_factory=lambda: ["train/loss", "val/loss"])  # Metrics to log
    log_generations_every_n_steps: int = 25  # The interval (in steps) at which to log generations

    spot_training: bool = False  # Whether to enable spot training
    # if true, will load from latest checkpoint if checkpoint dir with same
    # experiment name
    # else will create new checkpoint dir
    checkpoint_interval: int = 1000  # The interval (in steps) at which to save checkpoints
    best_metric: Optional[BestMetric] = None  # if set, save best checkpoint as well
    max_checkpoints_to_keep: int = 5  # The maximum number of checkpoints to keep
