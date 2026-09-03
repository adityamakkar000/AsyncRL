from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, PyTree
from omegaconf import MISSING

from src.data import DatasetConfig
from src.model import KVCache, ModelConfig
from src.workers.utils import MPQueues


@dataclass
class InferenceConfig:
    temperature: float = MISSING
    top_k: int | None = MISSING
    top_p: float | None = MISSING
    max_seq_len: int = MISSING
    max_decode_batch_size: int = MISSING
    tp: int = 1
    group_size: int = MISSING
    initial_sequence_len: int = 64
    kv_cache_dtype: str = "bfloat16"
    params_dtype: str = "bfloat16"
    reasoning_budget: int | None = None
    think_mode: bool = True
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

    def sub(self, new_batch: "InferenceState", index: Array | int) -> "InferenceState":
        return self.replace(  # type: ignore
            next_token=self.next_token.at[index].set(new_batch.next_token),
            kv_cache=[
                KVCache(
                    k=self.kv_cache[i].k.at[index].set(new_batch.kv_cache[i].k),
                    v=self.kv_cache[i].v.at[index].set(new_batch.kv_cache[i].v),
                    length=self.kv_cache[i].length.at[index].set(new_batch.kv_cache[i].length),
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

    def roll(self, index: Array) -> "InferenceState":
        """
        Roll a single-row state so its content starts at column 0.
        Args:
            index (Array): The cache length the row should have after rolling.
        Returns:
            InferenceState: The rolled state.
        """

        diff = jnp.reshape(index - self.kv_cache[0].length, ())
        return self.replace(  # type: ignore
            out_tokens=jnp.roll(self.out_tokens, diff, axis=1),
            out_logprobs=jnp.roll(self.out_logprobs, diff, axis=1),
            kv_cache=[
                KVCache(
                    k=jnp.roll(kv.k, diff, axis=1),
                    v=jnp.roll(kv.v, diff, axis=1),
                    length=jnp.reshape(index, (1,)),
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
    read_write_lock: Lock = field(default_factory=Lock)
    weight_iteration: int = 0


@dataclass
class ShardingConfig:
    opt_state_offload: bool = False
    data_shard_dim: int = 0
    cp_shard_dim: int = 0

    min_bytes_for_fsdp: int = int(1e6)  # 1e6/(1024*1024) = 1MB
    weight_shard_dim: int = 0

    dp_group_size: int = 1
    cp_group_size: int = 1
    fsdp_group_size: int = -1


@dataclass
class WandBConfig:
    project: str = "Debug"
    notes: str | None = None
    tags: list[str] | None = None


@dataclass
class LossConfig:
    rl_config: dict[str, Any] = MISSING
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    filter_zero_variance: bool = False


@dataclass
class AsyncConfig:
    train_workers: int = 1
    max_lag: int = 4


@dataclass
class EvalConfig:
    group_size: int = 8
    pass_k: list[int] = field(default_factory=lambda: [32, 64, 128])
    log_traces_n_prompts: int = -1
    eval_every_n_steps: int = 50
    eval_on_step_0: bool = False
    val_dataset_configs: list[DatasetConfig] = field(default_factory=list)


@dataclass
class TeacherConfig:
    teacher_checkpoint: str = MISSING
    teacher_step: int | None = None


@dataclass
class TrainerConfig:
    experiment_name: str = MISSING  # The name of the experiment
    train_dataset_config: DatasetConfig = MISSING  # The configuration for the data module
    model_config: ModelConfig = MISSING  # The configuration for the model
    loss_config: LossConfig = MISSING  # The configuration for the loss function

    eval_config: EvalConfig = field(default_factory=EvalConfig)

    async_config: AsyncConfig = field(default_factory=AsyncConfig)
    sharding_config: ShardingConfig = field(default_factory=ShardingConfig)

    teacher_config: TeacherConfig | None = None

    seed: int = 0

    num_steps: int = 1000
    train_batch_size: int = MISSING
    grad_accum_steps: int = 1

    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    weight_decay: float | None = None
    grad_clip: float | None = None

    # cosine lr
    learning_rate_init: float = 0.0
    learning_rate_peak: float = 1e-4
    learning_rate_end: float = 1e-5
    warmup_steps: float = 0.1  # fraction of total steps to use for learning rate warmup
    decay_steps: float = 0.9  # fraction of total steps to use for learning rate decay

    wandb_config: WandBConfig | None = None
    metrics_to_log: list[str] = field(default_factory=lambda: ["train/loss", "val/loss"])  # Metrics to log to terminal
    log_generations_every_n_steps: int = 25

    spot_training: bool = False  # Whether to enable spot training
    # if true, will load from latest checkpoint if checkpoint dir with same exp name
    # else will create new checkpoint dir

    checkpoint_interval: int = 1000
    max_checkpoints_to_keep: int = 5
    keep_every: int | None = None

    debug: bool = False


@dataclass
class AsyncOptions:
    train_workers: int
    inference_workers: int
    queues: MPQueues
    train_mesh: jax.sharding.Mesh
    inference_mesh: jax.sharding.Mesh
    global_mesh: jax.sharding.Mesh
