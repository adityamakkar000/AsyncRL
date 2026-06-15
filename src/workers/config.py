from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional, Protocol

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, PyTree
from omegaconf import MISSING

from src.data import DatasetConfig, RLBatch
from src.model import KVCache, ModelConfig


@dataclass
class InferenceConfig:
    temperature: float = MISSING
    top_k: Optional[int] = MISSING
    top_p: Optional[float] = MISSING
    max_seq_len: int = MISSING
    _max_decode_prompts: int = MISSING  # number of prompts to take during each continous run of the engine
    _max_decode_batch_size: int = MISSING  # at decode time how many samples to take for each device
    group_size: int = MISSING
    n_replicas: int = 1
    initial_sequence_len: int = 64
    kv_cache_dtype: str = "bfloat16"
    params_dtype: str = "bfloat16"
    reasoning_budget: Optional[int] = None
    think_mode: bool = True
    max_prefill_sequence_len: int = 1024
    system_prompt: bool = False


@struct.dataclass
class InferenceState:
    next_token: Array
    kv_cache: list[KVCache]
    key: Array
    seq_lens: Array
    stop_mask: Array
    end_of_think: Array
    out_tokens: Array
    out_logprobs: Array
    prompt_id: Array

    def sub(self, new_batch: "InferenceState", index: int) -> "InferenceState":
        return self.replace(
            next_token=self.next_token.at[index].set(new_batch.next_token),
            kv_cache=[
                KVCache(
                    k=self.kv_cache[i].k.at[index].set(new_batch.kv_cache[i].k),
                    v=self.kv_cache[i].v.at[index].set(new_batch.kv_cache[i].v),
                    length=self.kv_cache[i].length.copy(),
                )
                for i in range(len(self.kv_cache))
            ],
            key=self.key,
            seq_lens=self.seq_lens.at[index].set(new_batch.seq_lens),
            stop_mask=self.stop_mask.at[index].set(new_batch.stop_mask),
            end_of_think=self.end_of_think.at[index].set(new_batch.end_of_think),
            out_tokens=self.out_tokens.at[index].set(new_batch.out_tokens),
            out_logprobs=self.out_logprobs.at[index].set(new_batch.out_logprobs),
            prompt_id=self.prompt_id.at[index].set(new_batch.prompt_id),
        )

    def shift_batch(self) -> "InferenceState":
        return self.roll(jnp.max(self.seq_lens) - 1)

    def roll(self, index: int | Array) -> "InferenceState":
        """
        Roll the state to index
        Args:
            index (int | Array): The index to roll to.
        Returns:
            InferenceState: The rolled state.
        """

        diff = index - self.kv_cache[0].length
        return self.replace(  # type: ignore
            out_tokens=jnp.roll(self.out_tokens, diff, axis=1),
            out_logprobs=jnp.roll(self.out_logprobs, diff, axis=1),
            kv_cache=[
                KVCache(
                    k=jnp.roll(kv.k, diff, axis=1),
                    v=jnp.roll(kv.v, diff, axis=1),
                    length=index.copy(),  # type: ignore
                )
                for kv in self.kv_cache
            ],
        )


@dataclass
class InferenceShardings:
    mesh: jax.sharding.Mesh
    split_sharding: jax.NamedSharding
    replicate_sharding: jax.NamedSharding
    kv_cache_sharding: KVCache
    state_sharding: InferenceState
    params_sharding: PyTree
    prefill_shardings: dict[str, PyTree]
    decode_any_shardings: dict[str, PyTree]


@dataclass
class AsyncState:
    MRUparams: jax.Array
    updated: bool
    update_lock: Lock = field(default_factory=Lock)
    weight_iteration: int = 0


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


@dataclass
class RLConfig:
    algorithm: str = "grpo"
    epsilon_high: float = 1.0
    epsilon_low: float = 0.1
    filter_zero_variance: bool = False


@dataclass
class LossConfig:
    rl_config: RLConfig = field(default_factory=RLConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)


@dataclass
class AsyncConfig:
    train_workers: int = 1
    max_lag: int = 4


@dataclass
class TrainerConfig:
    experiment_name: str = MISSING  # The name of the experiment
    data_config: DatasetConfig = MISSING  # The configuration for the data module
    model_config: ModelConfig = MISSING  # The configuration for the model
    loss_config: LossConfig = MISSING  # The configuration for the loss function

    async_config: AsyncConfig = field(default_factory=AsyncConfig)  # The configuration for asynchronous training
    sharding_config: ShardingConfig = field(default_factory=ShardingConfig)  # The configuration for sharding

    # training config
    seed: int = 0

    num_steps: int = 1000
    grad_accum_steps: int = 1  # gradient accumulation steps

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
    log_generations_every_n_steps: int = 25

    spot_training: bool = False  # Whether to enable spot training
    # if true, will load from latest checkpoint if checkpoint dir with same exp name
    # else will create new checkpoint dir

    checkpoint_interval: int = 1000
    max_checkpoints_to_keep: int = 5
