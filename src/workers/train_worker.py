import json
import os
from collections import defaultdict
from functools import partial
from statistics import fmean
from typing import Any

import jax
import jax._src.distributed as dist
import numpy as np
import optax
import stax
from dotenv import load_dotenv
from einops import rearrange
from hydra.utils import instantiate
from jaxtyping import Array, PyTree
from omegaconf import OmegaConf
from stax import staxLogger as logger
from stax import sync_over_mesh
from stax.utils import metrics_all_reduce
from stax.writer import TableMetrics

from src.constants import RUN_PATH
from src.data import DataWorker, EvalWorker, RLBatch
from src.model import Model, make_weight_shard_fn

from .config import AsyncOptions, TrainerConfig
from .constants import EVAL_TIMEOUT, TIMEOUT
from .rdma_transfer import RDMATransferServer
from .utils import (
    PER_WORKER_SUMS,
    Key,
    memory_metrics,
    reduce_inference_metric,
    setup,
    to_steps,
    track_grad_norm,
    write_to_gcs,
)
from .worker import Worker

load_dotenv()


class AsyncTrainerWorker(Worker):
    """
    A Trainer class to handle the training process of a machine learning model using JAX on TPUs.
    """

    def __init__(self, config: TrainerConfig, async_options: AsyncOptions):
        """
        Initialize the training module.

        Args:
            config (TrainerConfig): Configuration object containing model, data, training parameters, etc.
            async_options (AsyncOptions): Configuration object containing asynchronous training parameters.
        """
        self.config = config
        self.async_options = async_options
        self.validate_config()

        self.log_info(f"Setting up {self.config.experiment_name}")
        self.log_info(f"runtime={stax.get_rank()}, distributed={dist.global_state.process_id}")

        with stax.Tracker(timer=True) as tracker:
            self._init_state()
            self._setup_model()
            self._setup_optimizer()
            self._setup_checkpointer()
            self._setup_functions()
            self._setup_dataset()
            self._setup_train_state()
            self._setup_writer()

            if not self.resumed:
                self.log_info("Saving intial checkpoint ...")
                self.save_checkpoint(step=0)
                dict_config = json.dumps(OmegaConf.to_container(self.config))
                write_to_gcs(self.config_path, dict_config)

            self.sync_train_workers("Trainer initialization")

        self.log_info(f"Trainer initialization complete in {tracker.data['time']:.2f} seconds")

    def validate_config(self):
        # validate config here
        cfg = self.config
        if cfg.grad_accum_steps < 1:
            raise ValueError("grad_accumulation must be at least 1")
        if cfg.num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if cfg.learning_rate_init < 0 or cfg.learning_rate_peak < 0 or cfg.learning_rate_end < 0:
            raise ValueError("learning rates must be non-negative")
        if cfg.optimizer not in ["adam", "adamw", "sgd"]:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
        if cfg.grad_clip is not None and cfg.grad_clip <= 0:
            raise ValueError("grad_clip must be positive if set")
        if to_steps(cfg.warmup_steps, cfg.num_steps) + to_steps(cfg.decay_steps, cfg.num_steps) > cfg.num_steps:
            raise ValueError("warmup_steps and decay_steps must sum to at most num_steps")
        if cfg.train_batch_size % cfg.loss_config.group_size != 0:
            raise ValueError(
                f"Batch size must be divisible by group size for proper batching in inference, got {cfg.train_batch_size} batch size and {cfg.loss_config.group_size} group size."
            )
        if cfg.model_config.fused_chunk_size is not None:
            flat_tokens = (cfg.train_batch_size // cfg.grad_accum_steps) * cfg.inference_config.max_seq_len
            if flat_tokens % cfg.model_config.fused_chunk_size != 0:
                raise ValueError(
                    f"Fused chunk size {cfg.model_config.fused_chunk_size} must evenly divide B * T {flat_tokens}."
                )

        n_hosts = jax.process_count()
        train_batch_size = cfg.train_batch_size
        assert train_batch_size % (cfg.loss_config.group_size * n_hosts) == 0, (
            f"Train batch size must be divisible by group size * number of hosts to get a correct number of prompts per batch for inference, got {train_batch_size} batch size, {cfg.loss_config.group_size} group size, and {n_hosts} hosts."
        )

        n_devices_train = jax.local_device_count() * cfg.async_config.train_workers
        assert train_batch_size % (n_devices_train * cfg.grad_accum_steps) == 0, (
            f"Train batch size must be divisible by train devices * grad_accum_steps for proper gradient accumulation, got {train_batch_size} train batch size, {n_devices_train} train devices, and {cfg.grad_accum_steps} grad_accum_steps."
        )
        assert (train_batch_size // cfg.loss_config.group_size) % jax.process_count() == 0, (
            f"Number of groups per step must be divisible by number of hosts for proper distribution of groups, got {train_batch_size} train batch size, {cfg.loss_config.group_size} group size, and {jax.process_count()} hosts."
        )

    @partial(setup, component="initialized state")
    def _init_state(self):
        self.train_fn = None

        self.model = None
        self.tx = None

        self.params = None
        self.teacher_params = None

        self.opt_state = None
        self.params_sharding = None
        self.opt_state_sharding = None
        self.shard_data_fn = None
        self.train_mesh = None

        self.best_checkpointer = None

        self.writer_id = None
        self.writer = None
        self.key = Key(self.config.seed)
        self.global_step = 0

        self.worker_rank = stax.get_rank()
        self.n_hosts: int = jax.process_count()
        self.n_devices = jax.device_count()
        gs_path = f"{RUN_PATH}/{self.config.experiment_name}"
        self.checkpoint_path = f"{gs_path}/checkpoints"
        self.profile_path = f"{gs_path}/profiles"
        self.config_path = f"{gs_path}/config.json"

        self.weight_iteration = 0
        self.local_mesh = jax.make_mesh(
            (jax.local_device_count(),),
            ("local_devices",),
            axis_types=(jax.sharding.AxisType.Explicit,),
            devices=np.array(jax.local_devices()),  # type: ignore
        )

        self.rdma_server = RDMATransferServer(train_workers=self.config.async_config.train_workers)

    @partial(setup, component="metric_logger")
    def _setup_writer(self):
        if writer_config := self.config.wandb_config:
            writer_kwargs: dict[str, Any] = {
                "metrics_to_print": self.config.metrics_to_log,
                "mesh": self.async_options.train_mesh,
                "lag_one_step": False,  # training blocks until params are done so log right away
            }
            if self.writer_id is not None:
                writer_kwargs["run_id"] = self.writer_id
            else:
                writer_kwargs["config"] = OmegaConf.to_object(self.config)

            writer = stax.WandBWriter(
                name=self.config.experiment_name,
                entity=os.getenv("WANDB_ENTITY", ""),
                project=writer_config.project,
                notes=writer_config.notes,
                tags=writer_config.tags,
                **writer_kwargs,
            )
        else:
            writer = stax.TextWriter(
                metrics_to_print=self.config.metrics_to_log,
                mesh=self.async_options.train_mesh,
                lag_one_step=False,
            )

        self.writer_id = writer.id
        self.writer = writer

    @partial(setup, component="train and val functions")
    def _setup_functions(self):
        assert self.tx is not None, "self.tx is None"
        assert self.model is not None, "self.model is None"

        abstract_state = self.model.init_state(jax.random.PRNGKey(0), tx=self.tx, sharding=None)
        params_shape, opt_state_shape = abstract_state["params"], abstract_state["opt_state"]

        self.loss_fn = instantiate(
            self.config.loss_config.rl_config, _recursive_=False, loss_config=self.config.loss_config
        )

        # val fn not needed since we just care about val reward, not loss
        self.train_fn, _val_fn, shardings = stax.fn.get_steps_fn(
            self.loss_fn,
            self.model,
            self.tx,
            has_aux=True,
            grad_steps=self.config.grad_accum_steps,
            reduce_fn=self.loss_fn.compute_normalization,
            val_steps=0,  # val steps is not used
            sharding=stax.ShardingConfig(
                params_shape=params_shape,
                opt_state_shape=opt_state_shape,
                mesh_config=stax.MeshConfig(
                    dp=self.config.sharding_config.dp_group_size,
                    fsdp=self.config.sharding_config.fsdp_group_size,
                    cp_ulysses=self.config.sharding_config.cp_group_size,
                ),
                opt_state_offload=self.config.sharding_config.opt_state_offload,
                min_bytes_for_fsdp=self.config.sharding_config.min_bytes_for_fsdp,
                data_shard_dim=self.config.sharding_config.data_shard_dim,
                cp_shard_dim=self.config.sharding_config.cp_shard_dim,
                weight_shard_fn=make_weight_shard_fn(self.config.sharding_config.weight_shard_dim),
            ),
            # only use devices on the train worker's mesh
            devices=self.async_options.train_mesh.devices.reshape(-1),
            # jit kwargs
            train_jit_kwargs={"donate_argnums": (1,)},
        )

        self.params_sharding, self.opt_state_sharding, self.shard_data_fn = (
            shardings.param_sharding,
            shardings.opt_state_sharding,
            shardings.shard_data,
        )

        self.train_mesh = shardings.mesh
        self.shardings = shardings
        dtype = self.config.inference_config.params_dtype
        self.replicate_params_fn = jax.jit(
            lambda params: jax.tree.map(lambda p: p.astype(dtype), params),
            out_shardings=jax.NamedSharding(self.train_mesh, jax.P()),
        )

    @partial(setup, component="dataset")
    def _setup_dataset(self):
        assert self.model is not None, "Model must be set up before the data workers."
        group_size = self.config.loss_config.group_size
        queues = self.async_options.queues
        train_workers = self.async_options.train_workers

        self.data_worker = DataWorker(
            self.config.train_envs,
            queues,
            rank=self.worker_rank,
            seed=self.config.seed,
            n_prompts=self.config.train_batch_size // group_size // train_workers,
            group_size=group_size,
            pad_token=self.model.pad_token,
            max_seq_len=self.config.inference_config.max_seq_len,
            max_lag=self.config.async_config.max_lag,
            filter_zero_variance=self.config.loss_config.filter_zero_variance,
            timeout=TIMEOUT,
        )

        self.eval_worker = EvalWorker(
            self.config.eval_config,
            queues,
            n_train_workers=train_workers,
            rank=self.worker_rank,
            timeout=EVAL_TIMEOUT,
        )

    @partial(setup, component="model")
    def _setup_model(self):
        self.model = Model(self.config.model_config)

    @partial(setup, component="train state")
    def _setup_train_state(self):
        assert self.model is not None, "Model must be set up before train state init."
        assert self.tx is not None, "Optimizer must be set up before train state init."
        assert self.params_sharding is not None and self.opt_state_sharding is not None, (
            "Sharding must be set up before train state init."
        )
        assert self.checkpointer is not None, "checkpointer must be set up before initializing train state"

        if self.config.teacher_config is not None:
            logger.info("Loading teacher/reference model", log_for_all=True)
            _teacher_step, teacher_params, _teacher_metadata = self.model.load_from_ckpt(
                self.config.teacher_config.teacher_checkpoint, self.config.teacher_config.teacher_step
            )
            self.teacher_params = jax.tree.map(lambda x, s: jax.device_put(x, s), teacher_params, self.params_sharding)

        if self.resumed:
            self.log_info("Checkpoint found, skipping parameter initialization.")
            self.restore_save_tree()
            return

        self.log_info("Initializing new run ...")
        out_state = self.model.init_state(rng=self.key(), tx=self.tx, sharding=self.shardings)

        self.params: Any = out_state["params"]
        self.opt_state = out_state["opt_state"]

        logger.info(
            f"Params intialized with total size: {self.model.count_params(self.params):_} parameters.", log_for_all=True
        )

    @partial(setup, component="optimizer")
    def _setup_optimizer(self):
        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=self.config.learning_rate_init,
            peak_value=self.config.learning_rate_peak,
            warmup_steps=to_steps(self.config.warmup_steps, self.config.num_steps),
            decay_steps=to_steps(self.config.decay_steps, self.config.num_steps),
            end_value=self.config.learning_rate_end,
        )

        match self.config.optimizer:
            case "adam":
                optimizer = optax.adam
            case "adamw":
                optimizer = optax.adamw
            case "sgd":
                optimizer = optax.sgd
            case _:
                raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        clip = (
            optax.clip_by_global_norm(self.config.grad_clip) if self.config.grad_clip is not None else optax.identity()
        )
        optimizer_args: dict = {"learning_rate": lr_scheduler, **self.config.optimizer_kwargs}
        self.tx = optax.chain(track_grad_norm(), clip, optax.inject_hyperparams(optimizer)(**optimizer_args))

    @partial(setup, component="checkpointer")
    def _setup_checkpointer(self):
        self.log_info(f"rank: {stax.get_rank()}")

        train_mesh = self.async_options.train_mesh
        self.checkpointer = stax.Checkpointer(
            output_dir=self.checkpoint_path,
            max_to_keep=self.config.max_checkpoints_to_keep,
            # only allow train workers to write checkpoints
            active_processes={d.process_index for d in train_mesh.devices.flat},
            train_mesh=train_mesh,
            keep_every=self.config.keep_every,
        )

    def make_save_tree(
        self,
        *,
        params: PyTree | None = None,
        opt_state: PyTree | None = None,
        metadata_metrics: dict[str, float] | None = None,
    ):
        assert self.train_mesh is not None, "train mesh should be intialized for checkpointing"

        state = {
            "params": self.params if params is None else params,
            "opt_state": self.opt_state if opt_state is None else opt_state,
        }

        metadata = {
            "writer_id": self.writer_id,
            "global_step": self.global_step,
            "key": np.asarray(self.key.key).tolist(),
        }
        if self.worker_rank == 0:
            metadata["dataset"] = self.data_worker.save()

        def convert_metric(x):
            return x.item() if isinstance(x, Array) else x

        if metadata_metrics is not None:
            metadata_metrics = jax.tree.map(convert_metric, metadata_metrics)
            metadata |= metadata_metrics

        return state, metadata

    def save_checkpoint(self, step: int):
        if not self.config.debug:
            assert self.checkpointer is not None, "Checkpointer not set up."
            state, metadata = self.make_save_tree()
            self.sync_train_workers(f"dataset_snapshot_{step}")
            self.log_info(f"[checkpointer] Saving checkpoint at step {step} ...")
            self.checkpointer.save_checkpoint(step, save_tree=state, metadata=metadata)
        else:
            self.log_info(f"Debug mode enabled, skipping checkpoint save at step {step}.")

    def restore_save_tree(self):
        assert self.checkpointer is not None, "Checkpointer not set up."
        assert self.model is not None, "Model not set up."
        assert self.tx is not None, "Optimizer not set up."

        if self.checkpointer.latest_step is None:
            raise ValueError("No checkpoint found to restore from.")

        self.global_step = self.checkpointer.latest_step

        def with_sharding(x, s):
            return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s)

        abstract_state = self.model.init_state(jax.random.PRNGKey(0), tx=self.tx)
        restored = self.checkpointer.restore(
            state={
                "params": jax.tree.map(with_sharding, abstract_state["params"], self.params_sharding),
                "opt_state": jax.tree.map(with_sharding, abstract_state["opt_state"], self.opt_state_sharding),
            }
        )
        state, metadata = restored["state"], restored["metadata"]
        self.params = state["params"]
        self.opt_state = state["opt_state"]

        self.key.key = jax.numpy.asarray(metadata["key"], dtype=jax.numpy.uint32)
        if self.worker_rank == 0:
            self.data_worker.restore(metadata["dataset"])

        self.writer_id = metadata.get("writer_id", None)

    def train_sync_weights(self):
        assert self.train_mesh is not None, "Train mesh must be set up to sync weights."

        params_cast = self.replicate_params_fn(self.params)
        if self.worker_rank == 0:
            for _ in range(self.async_options.inference_workers):
                self.async_options.queues.weight_sync_queue.put("sync")

        _ = self.rdma_server.transfer({"params": params_cast}, init_mesh=self.train_mesh)

        del params_cast
        self.weight_iteration += 1

    def get_step_metrics(
        self,
        train_metrics: dict,
        local_rollout_metrics: dict,
        local_batch_metrics: dict,
        *,
        prep_batch_time: float,
        learner_time: float,
        sync_time: float,
        step_time: float,
        eval_out: tuple[dict[str, float], list[TableMetrics]] | None,
        eval_time: float | None,
    ) -> tuple[dict, list[TableMetrics]]:
        assert self.opt_state is not None, "Train state not initialized."

        metrics = (
            train_metrics
            | metrics_all_reduce(local_rollout_metrics | local_batch_metrics, self.async_options.train_mesh)
            | {f"devices/{k}": v for k, v in memory_metrics(jax.local_devices()).items()}
            | {
                "train/lr": self.opt_state[2].hyperparams["learning_rate"],
                "train/grad_norm": self.opt_state[0].grad_norm,
                "train/prep_batch_time": prep_batch_time,
                "train/learner_step_time": learner_time,
                "train/weight_sync_time": sync_time,
                "train/step_time": step_time,
            }
        )
        tables: list[TableMetrics] = []

        if stax.get_rank() == 0:
            metrics |= self.get_inference_metrics()

        if eval_out is not None:
            eval_metrics, eval_tables = eval_out
            metrics |= eval_metrics | {"eval/time": eval_time}
            tables.extend(eval_tables)

        unstable = metrics.get("train/is_ratio") and abs(1 - metrics["train/is_ratio"].item()) > 0.05
        if self.global_step % self.config.log_generations_every_n_steps == 0 or unstable:
            if unstable:
                logger.warning(f"unstable training detected, saving table for step {self.global_step}")
            tables.append(TableMetrics("train/generations", self.data_worker.table()))

        return metrics, tables

    def get_inference_metrics(self) -> dict[str, float]:
        by_worker: dict[int, list[dict]] = defaultdict(list)
        while not self.async_options.queues.inference_metrics_queue.empty():
            record = self.async_options.queues.inference_metrics_queue.get(timeout=TIMEOUT)
            by_worker[record["worker_id"]].append(record)

        metrics: dict[str, float] = {}
        per_key: dict[str, list[float]] = defaultdict(list)

        for worker_id, records in by_worker.items():
            for k in {k for r in records for k in r} - {"worker_id"}:
                mean = (sum if k in PER_WORKER_SUMS else fmean)([r[k] for r in records if k in r])
                metrics[f"inference/worker_{worker_id}/{k}"] = mean
                per_key[k].append(mean)

        for k, values in per_key.items():
            metrics[f"inference/{k}"] = reduce_inference_metric(k, values)

        expected_workers = self.async_options.inference_workers * jax.local_device_count()
        metrics["inference/workers_reporting"] = len(by_worker)

        if len(by_worker) < expected_workers:
            logger.warning(
                f"Only {len(by_worker)}/{expected_workers} inference workers reported metrics.",
                log_for_all=True,
            )

        return metrics

    def global_train_batch(self, local_batch):
        assert self.shard_data_fn is not None, "Sharding function not set up."
        global_batch = self.shard_data_fn(local_batch)
        return jax.tree.map(
            lambda x: rearrange(
                x,
                "(m g) ... -> g m ...",
                m=self.config.train_batch_size // self.config.grad_accum_steps,
                g=self.config.grad_accum_steps,
            ),  # [grad_accum_steps, minibatch_size, seq_len]
            global_batch,
        )

    def train_step(
        self, params, opt_state, local_batch, teacher_params=None, profile=False
    ) -> tuple[PyTree, PyTree, dict]:
        assert self.train_fn is not None, "Train function not set up."

        trace_path = f"{self.profile_path}/step_{self.global_step}" if profile else None
        with stax.Tracker(trace=trace_path):
            global_train_batch = self.global_train_batch(local_batch)
            out = self.train_fn(params, opt_state, global_train_batch, teacher_params=teacher_params)
            # we have to sync weights so might as well block to get true step time
            jax.tree.map(lambda x: x.block_until_ready(), out)

        return out["params"], out["opt_state"], out["metrics"]

    def compile_train_fn(self):
        assert self.train_fn is not None, "Train function not set up."
        self.log_info("Precompiling test batch")

        local_test_batch = RLBatch.get_test_batch(
            batch_size=self.config.train_batch_size // self.config.async_config.train_workers,
            max_seq_len=self.config.inference_config.max_seq_len,
        )
        self.train_fn.lower(  # type: ignore
            self.params, self.opt_state, self.global_train_batch(local_test_batch), teacher_params=self.teacher_params
        ).compile()

    def train(self):
        assert self.params is not None and self.opt_state is not None, "Train state not initialized."
        assert self.shard_data_fn is not None, "Sharding function not set up."
        assert self.model is not None, "Model not set up."
        assert self.writer is not None, "Writer not set up."
        assert self.checkpointer is not None, "Checkpointer not set up."

        self.train_sync_weights()
        self.data_worker.start()
        self.compile_train_fn()

        initial_step = self.global_step
        self.log_info(f"Starting training loop at step {self.global_step}")
        while self.global_step < self.total_steps:
            with stax.Tracker(timer=True) as step_t:
                with stax.Tracker(timer=True) as prep_batch_t:
                    local_train_batch, local_rollout_metrics, local_batch_metrics = self.data_worker.next(
                        self.weight_iteration
                    )

                with stax.Tracker(timer=True) as learner_t:
                    self.params, self.opt_state, train_metrics = self.train_step(
                        self.params,
                        self.opt_state,
                        local_train_batch,
                        teacher_params=self.teacher_params,
                        profile=self.global_step == initial_step,
                    )

                with stax.Tracker(timer=True) as sync_t:
                    self.train_sync_weights()

            if self.should_eval:
                with stax.Tracker(timer=True) as eval_t:
                    eval_out = self.eval_worker.run_eval()

            metrics, tables = self.get_step_metrics(
                train_metrics,
                local_rollout_metrics,
                local_batch_metrics,
                prep_batch_time=prep_batch_t.data["time"],
                learner_time=learner_t.data["time"],
                sync_time=sync_t.data["time"],
                step_time=step_t.data["time"],
                eval_out=eval_out if self.should_eval else None,
                eval_time=eval_t.data["time"] if self.should_eval else None,
            )

            self.writer(self.global_step, metrics, table_metrics=tables)
            self.global_step += 1

            # save after you update state since if you want to save every 10 steps
            # you want to save after you have done 10 steps and resume at the 11th step
            if self.global_step % self.config.checkpoint_interval == 0 or self.global_step == self.total_steps:
                self.save_checkpoint(step=self.global_step)

        self.log_info("Training complete.")

    def start(self):
        try:
            self.train()
        finally:
            self.finish()

    def sync_train_workers(self, name: str):
        sync_over_mesh(name, self.async_options.train_mesh)

    def block_checkpointer(self):
        assert self.checkpointer is not None, "Checkpointer not set up."
        self.checkpointer.wait_until_finished()

    @partial(setup, component="cleanup")
    def finish(self):
        if self.checkpointer:
            self.log_info("[checkpointer] saving final checkpointing...")
            self.block_checkpointer()
        self.data_worker.cleanup()
        self.eval_worker.cleanup()
        if self.worker_rank == 0:
            for _ in range(self.async_options.inference_workers):
                self.async_options.queues.weight_sync_queue.put("stop")

    @property
    def should_eval(self) -> bool:
        return self.global_step % self.config.eval_config.eval_every_n_steps == 0 and (
            self.global_step > 0 or self.config.eval_config.eval_on_step_0
        )

    @property
    def total_steps(self):
        return self.config.num_steps

    @property
    def resumed(self):
        assert self.checkpointer is not None, "Checkpointer not set up."
        return self.checkpointer.latest_step is not None

    def log_info(self, msg: str):
        logger.info(msg, log_for_all=self.config.debug)
