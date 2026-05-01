from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from jaxtyping import Array
from omegaconf import MISSING

from src.data import DataConfig, RLBatch
from src.inference_engine import InferenceConfig
from src.model import ModelConfig


class LossFunction(Protocol):
    """
    A protocol for loss functions used in RLVR.
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
class AnnealedLoss: ...


@dataclass
class RLConfig:
    algorithm: str = "grpo"
    epsilon_high: float = 1.0
    epsilon_low: float = 0.1


@dataclass
class LossConfig:
    rl_config: RLConfig = field(default_factory=RLConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    annealing_config: Optional[AnnealedLoss] = None  # TODO: Annealed RL config will go here


@dataclass
class AsyncConfig:
    train_workers: int = 1
    max_prompt_queue_size: int = 4  # multiple of num of prompts to keep


@dataclass
class TrainerConfig:
    experiment_name: str = MISSING  # The name of the experiment
    data_config: DataConfig = MISSING  # The configuration for the data module
    model_config: ModelConfig = MISSING  # The configuration for the model
    loss_config: LossConfig = MISSING  # The configuration for the loss function

    async_config: AsyncConfig = field(default_factory=AsyncConfig)  # The configuration for asynchronous training
    sharding_config: ShardingConfig = field(default_factory=ShardingConfig)  # The configuration for sharding

    # training config
    seed: int = 0

    num_steps: int = 1000
    grad_accum_steps: int = 1  # gradient accumulation steps
    val_interval: int = 100

    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    weight_decay: Optional[float] = None  # The weight decay coefficient
    grad_clip: Optional[float] = None  # The maximum gradient norm for clipping

    # cosine lr
    learning_rate_init: float = 0.0  # The initial learning rate
    learning_rate_peak: float = 1e-4  # The maximum learning rate
    learning_rate_end: float = 1e-5  # The final learning rate
    warmup_steps: float = 0.1  # The fraction of total steps to use for learning rate warmup
    decay_steps: float = 0.9  # The fraction of total steps to use for learning rate decay

    wandb_config: Optional[WandBConfig] = None
    metrics_to_log: List[str] = field(default_factory=lambda: ["train/loss", "val/loss"])  # Metrics to log to terminal
    log_generations_every_n_steps: int = 25  # The interval (in steps) at which to log generations

    spot_training: bool = False  # Whether to enable spot training
    # if true, will load from latest checkpoint if checkpoint dir with same exp name
    # else will create new checkpoint dir

    checkpoint_interval: int = 1000
    max_checkpoints_to_keep: int = 5
