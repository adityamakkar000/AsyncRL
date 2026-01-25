from dataclasses import dataclass, field
from typing import List, Literal, Optional

from omegaconf import MISSING

from src.data import DataConfig
from src.model import ModelConfig


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


@dataclass
class AnnealedLoss:
    # TODO: represent the real experiment
    ...


@dataclass
class BestMetric:
    name: str = MISSING  # Name of the metric to monitor
    maximize: bool = False  # Whether to maximize or minimize the metric


@dataclass
class TrainerConfig:
    experiment_name: str = MISSING  # The name of the experiment
    data_config: DataConfig = MISSING  # The configuration for the data module
    model_config: ModelConfig = MISSING  # The configuration for the model

    sharding_config: ShardingConfig = field(default_factory=ShardingConfig)  # The configuration for sharding

    # training config
    seed: int = 0  # The random seed to use for reproducibility
    # TODO:
    # figure out if we want epochs or steps
    # pros of epochs: more interpretable, cons: lr scheduling, etc
    num_steps: int = 1000  # The number of training epochs
    grad_steps: int = 1  # The number of gradient accumulation steps
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

    spot_training: bool = False  # Whether to enable spot training
    # if true, will load from latest checkpoint if checkpoint dir with same
    # experiment name
    # else will create new checkpoint dir
    checkpoint_interval: int = 1000  # The interval (in steps) at which to save checkpoints
    best_metric: Optional[BestMetric] = None  # if set, save best checkpoint as well
    max_checkpoints_to_keep: int = 5  # The maximum number of checkpoints to keep

    