from dataclasses import dataclass, field
from threading import Condition, Event, Lock
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, PyTree
from omegaconf import MISSING

from src.data import EnvConfig
from src.model import KVCache, ModelConfig
from src.workers.utils import MPQueues


@dataclass
class InferenceConfig:
    temperature: float = MISSING
    top_k: int | None = MISSING
    top_p: float | None = MISSING
    max_seq_len: int = MISSING
    max_decode_batch_size: int = MISSING
    prefill_multiplier: float = 2.0
    warmup_seq_len: int = 1024
    tp: int = 1
    group_size: int = MISSING
    min_prefill_length: int = 128
    kv_cache_dtype: str = "bfloat16"
    params_dtype: str = "bfloat16"
    reasoning_budget: int | None = None
    think_mode: bool = True


class InferenceState(struct.PyTreeNode):
    next_token: Array
    kv_cache: KVCache
    key: Array
    seq_lens: Array
    stop_mask: Array
    end_of_think: Array
    out_tokens: Array
    out_logprobs: Array
    prompt_id: Array

    def slice(self, index: Array | int) -> "InferenceState":
        def slice_arg(arr: Array, axis: int = 0):
            return jax.lax.dynamic_slice_in_dim(arr, start_index=index, slice_size=1, axis=axis)

        return InferenceState(
            next_token=slice_arg(self.next_token),
            kv_cache=KVCache(
                k=jax.tree.map(lambda x: slice_arg(x, axis=1), self.kv_cache.k),
                v=jax.tree.map(lambda x: slice_arg(x, axis=1), self.kv_cache.v),
                length=slice_arg(self.kv_cache.length),
            ),
            key=self.key,
            seq_lens=slice_arg(self.seq_lens),
            stop_mask=slice_arg(self.stop_mask),
            end_of_think=slice_arg(self.end_of_think),
            out_tokens=slice_arg(self.out_tokens),
            out_logprobs=slice_arg(self.out_logprobs),
            prompt_id=slice_arg(self.prompt_id),
        )

    def sub(self, batch: "InferenceState", replace_index: Array | int) -> "InferenceState":
        new_batch = batch.slice(replace_index)
        index = jnp.argmax(self.stop_mask[:, 0])

        def replace(arr: Array, new_arr: Array, axis: int = 0) -> Array:
            return jax.lax.dynamic_update_slice_in_dim(arr, new_arr.astype(arr.dtype), index, axis=axis)

        return InferenceState(
            next_token=replace(self.next_token, new_batch.next_token),
            kv_cache=KVCache(
                k=jax.tree.map(lambda x, y: replace(x, y, axis=1), self.kv_cache.k, new_batch.kv_cache.k),
                v=jax.tree.map(lambda x, y: replace(x, y, axis=1), self.kv_cache.v, new_batch.kv_cache.v),
                length=replace(self.kv_cache.length, new_batch.kv_cache.length),
            ),
            key=self.key,
            seq_lens=replace(self.seq_lens, new_batch.seq_lens),
            stop_mask=replace(self.stop_mask, new_batch.stop_mask),
            end_of_think=replace(self.end_of_think, new_batch.end_of_think),
            out_tokens=replace(self.out_tokens, new_batch.out_tokens),
            out_logprobs=replace(self.out_logprobs, new_batch.out_logprobs),
            prompt_id=replace(self.prompt_id, new_batch.prompt_id),
        )

    @classmethod
    def get_default(
        cls, batch_size: int, seq_len: int, key: Array, kv_cache: KVCache, sharding: jax.NamedSharding
    ) -> "InferenceState":
        return cls(
            next_token=jnp.zeros((batch_size, 1), dtype=jnp.int32, out_sharding=sharding),
            kv_cache=kv_cache,
            key=key,
            seq_lens=jnp.zeros((batch_size,), dtype=jnp.int32, out_sharding=sharding),
            stop_mask=jnp.ones((batch_size, 1), dtype=jnp.bool, out_sharding=sharding),
            end_of_think=jnp.zeros((batch_size, 1), dtype=jnp.bool, out_sharding=sharding),
            out_tokens=jnp.zeros((batch_size, seq_len), dtype=jnp.int32, out_sharding=sharding),
            out_logprobs=jnp.zeros((batch_size, seq_len), dtype=jnp.float32, out_sharding=sharding),
            prompt_id=jnp.zeros((batch_size, 1), dtype=jnp.int32, out_sharding=sharding),
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
    weight_sync_event: Event = field(default_factory=Event)
    n_workers_ready: int = 0
    n_workers_condition: Condition = field(default_factory=Condition)
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
    eval_envs: dict[str, EnvConfig] = field(default_factory=dict)


@dataclass
class TeacherConfig:
    teacher_checkpoint: str = MISSING
    teacher_step: int | None = None


@dataclass
class TrainerConfig:
    experiment_name: str = MISSING  # The name of the experiment
    train_envs: dict[str, EnvConfig] = MISSING
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
    optimizer_kwargs: dict = field(default_factory=dict)
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
