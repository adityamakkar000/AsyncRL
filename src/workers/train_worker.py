import json
import os
import threading
import time
from functools import partial
from typing import Any, Optional

import jax
import jax._src.distributed as dist
import numpy as np
import optax
import stax
from dotenv import load_dotenv
from einops import rearrange
from jaxtyping import Array, PyTree
from omegaconf import OmegaConf
from stax import staxLogger as logger
from stax import sync_over_mesh
from stax.utils import metrics_all_reduce

from src.constants import CHECKPOINTS, GS_BUCKET, PROFILE, TIMEOUT, AsyncOptions
from src.data import DataLoader, InferenceRollout, RLBatch
from src.model import Model

from .config import TrainerConfig
from .loss import get_single_step
from .utils import Key, get_current_vm_internal_ip, setup, setup_transfer_server, write_to_gcs
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

        logger.info(f"Setting up {self.config.experiment_name}", log_for_all=True)
        logger.info(f"runtime={stax.get_rank()}, distributed={dist.global_state.process_id}", log_for_all=True)

        self.ip = get_current_vm_internal_ip()
        self.transfer_server = setup_transfer_server(self.ip, port=8000)

        with stax.Tracker(timer=True) as tracker:
            self._init_state()
            self._setup_model()
            self._setup_optimizer()
            self._setup_checkpointer()
            self._setup_functions()
            self._setup_dataset()
            self._setup_train_state()
            self._setup_writer()
            self._start_queue()

            if not self.resumed:
                logger.info("Saving intial checkpoint ...", log_for_all=True)
                self.save_checkpoint(step=0)
                dict_config = json.dumps(OmegaConf.to_container(self.config))
                config_path = f"{self.gs_path}/config.json"
                write_to_gcs(config_path, dict_config)

            self.train_sync_weights()
            self.sync_train_workers("Trainer initialization")

        logger.info(f"Trainer initialization complete in {tracker.data['time']:.2f} seconds", log_for_all=True)

    def validate_config(self):
        """Method to validate the TrainerConfig parameters."""
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
        if (cfg.warmup_steps + cfg.decay_steps) > 1.0:
            raise ValueError("warmup_steps and decay_steps must sum to at most 1.0")
        if cfg.sharding_config.sharding_type not in ["single", "dp", "fsdp"]:
            raise ValueError("sharding_type must be one of 'single', 'dp', or 'fsdp'")
        if cfg.data_config.batch_size % cfg.loss_config.inference_config.group_size != 0:
            raise ValueError(
                f"Batch size must be divisible by group size for proper batching in inference, got {cfg.data_config.batch_size} batch size and {cfg.loss_config.inference_config.group_size} group size."
            )

        n_hosts = jax.process_count()
        train_batch_size = cfg.data_config.batch_size
        assert train_batch_size % (cfg.loss_config.inference_config.group_size * n_hosts) == 0, (
            f"Train batch size must be divisible by group size * number of hosts to get a correct number of prompts per batch for inference, got {train_batch_size} batch size, {cfg.loss_config.inference_config.group_size} group size, and {n_hosts} hosts."
        )

        n_devices = jax.device_count() if cfg.sharding_config.sharding_type in ["fsdp", "dp"] else 1

        if cfg.sharding_config.sharding_type == "single":
            assert jax.process_count() == 1, (
                "Single sharding type does not support distributed training across multiple hosts."
            )

        assert train_batch_size % (n_devices * cfg.grad_accum_steps) == 0, (
            f"Train batch size must be divisible by number of devices * grad_accum_steps for proper gradient accumulation, got {train_batch_size} train batch size, {n_devices} devices, and {cfg.grad_accum_steps} grad_accum_steps."
        )

        assert (train_batch_size // cfg.loss_config.inference_config.group_size) % jax.process_count() == 0, (
            f"Number of groups per step must be divisible by number of hosts for proper distribution of groups, got {train_batch_size} train batch size, {cfg.loss_config.inference_config.group_size} group size, and {jax.process_count()} hosts."
        )

    @partial(setup, component="initialized state")
    def _init_state(self):
        self.train_fn = None

        self.model = None
        self.tx = None

        self.params = None
        self.opt_state = None
        self.params_sharding = None
        self.opt_state_sharding = None
        self.shard_data_fn = None
        self.train_mesh = None

        self.train_dataset = None

        self.checkpointer = None
        self.best_checkpointer = None

        self.writer_id = None
        self.writer = None
        self.key = Key(self.config.seed)
        self.global_step = 0

        self.worker_rank = stax.get_rank()
        self.n_hosts = jax.process_count()
        self.n_devices = jax.device_count()
        self.gs_path = f"{GS_BUCKET}/runs/{self.config.experiment_name}"

        self.weight_iteration = 0
        self.local_mesh = jax.make_mesh(
            (jax.local_device_count(),),
            ("local_devices",),
            axis_types=(jax.sharding.AxisType.Explicit,),
            devices=np.array(jax.local_devices()),
        )

    @partial(setup, component="metric_logger")
    def _setup_writer(self):
        if writer_config := self.config.wandb_config:
            writer_kwargs: dict[str, Any] = {
                "metrics_to_print": self.config.metrics_to_log,
                "mesh": self.async_options.train_mesh,
            }
            if self.writer_id is not None:
                writer_kwargs["run_id"] = self.writer_id
            else:
                writer_kwargs["config"] = OmegaConf.to_object(self.config)

            writer = stax.WandBWriter(
                name=self.config.experiment_name,
                entity=os.getenv("WANDB_ENTITY", ""),
                project=writer_config.project,
                **writer_kwargs,
            )
        else:
            writer = stax.TextWriter(metrics_to_print=self.config.metrics_to_log)

        self.writer_id = writer.id
        self.writer = writer

    @partial(setup, component="train and val functions")
    def _setup_functions(self):
        """Setup training and validation functions."""
        assert self.tx is not None, "self.tx is None"
        assert self.model is not None, "self.model is None"

        abstract_state = self.model.init_state(jax.random.PRNGKey(0), tx=self.tx, abstract=True)
        params_shape, opt_state_shape = abstract_state["params"], abstract_state["opt_state"]

        step_fn = get_single_step(self.config.loss_config)

        # val fn not needed since we just care about val reward, not loss
        self.train_fn, _val_fn, shardings = stax.fn.get_steps_fn(
            step_fn,
            self.model,
            self.tx,
            has_aux=True,
            grad_steps=self.config.grad_accum_steps,
            val_steps=0,  # val steps is not used
            sharding=stax.ShardingConfig(
                params_shape=params_shape,
                opt_state_shape=opt_state_shape,
                sharding_type=stax.ShardingType(self.config.sharding_config.sharding_type),
                opt_state_offload=self.config.sharding_config.opt_state_offload,
                min_bytes_for_fsdp=self.config.sharding_config.min_bytes_for_fsdp,
                data_shard_dim=self.config.sharding_config.data_shard_dim,
                weight_shard_dim=self.config.sharding_config.weight_shard_dim,
            ),
            # only use devices on the train worker's mesh
            devices=self.async_options.train_mesh.devices.reshape(-1),
        )

        self.params_sharding, self.opt_state_sharding, self.shard_data_fn = (
            shardings.param_sharding,
            shardings.opt_state_sharding,
            shardings.shard_data,
        )

        self.train_mesh = shardings.mesh

    @partial(setup, component="dataset")
    def _setup_dataset(self):
        self.train_n_prompts: int = self.config.data_config.batch_size // (
            self.config.loss_config.inference_config.group_size
        )

        self.train_n_prompts_per_host = self.train_n_prompts // self.async_options.train_workers

        max_seq_length = self.config.loss_config.inference_config.max_seq_len
        hf_model = self.config.model_config.hf_model_name
        self.train_dataset = DataLoader(
            self.config.data_config, max_seq_length, hf_model, self.async_options.train_mesh
        )

    @partial(setup, component="model")
    def _setup_model(self):
        """Setup the model for training."""
        self.model = Model(self.config.model_config)

    @partial(setup, component="train state")
    def _setup_train_state(self):
        assert self.model is not None, "Model must be set up before train state init."
        assert self.tx is not None, "Optimizer must be set up before train state init."
        assert self.params_sharding is not None and self.opt_state_sharding is not None, (
            "Sharding must be set up before train state init."
        )
        assert self.checkpointer is not None, "checkpointer must be set up before initializing train state"

        if self.resumed:
            logger.info(
                "Spot training enabled and checkpoint found, skipping parameter initialization.", log_for_all=True
            )
            self.restore_save_tree()
            return

        logger.info("Initializing new run ...", log_for_all=True)
        sharding = {"params": self.params_sharding, "opt_state": self.opt_state_sharding}
        out_state = self.model.init_state(rng=self.key(), tx=self.tx, sharding=sharding, abstract=False)
        self.params = out_state["params"]
        self.opt_state = out_state["opt_state"]

        logger.info(
            f"Params intialized with total size: {self.model.count_params(self.params):_} parameters.", log_for_all=True
        )

    @partial(setup, component="optimizer")
    def _setup_optimizer(self):
        """Setup the optimizer and learning rate scheduler."""
        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=self.config.learning_rate_init,
            peak_value=self.config.learning_rate_peak,
            warmup_steps=int(self.config.warmup_steps * self.config.num_steps),
            decay_steps=int(self.config.decay_steps * self.config.num_steps),
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

        optimizer_args: dict = {
            "learning_rate": lr_scheduler,
        }
        if self.config.weight_decay is not None and self.config.optimizer == "adamw":
            optimizer_args["weight_decay"] = self.config.weight_decay
            optimizer_args["eps"] = 1e-15
        if self.config.optimizer == "adam" or self.config.optimizer == "adamw":
            optimizer_args["mu_dtype"] = "float32"
        self.tx = optax.chain(
            clip,
            optax.inject_hyperparams(optimizer)(
                **optimizer_args,
            ),
        )

    @partial(setup, component="checkpointer")
    def _setup_checkpointer(self):
        """Setup checkpointing mechanism."""

        path = f"{self.gs_path}/{CHECKPOINTS}/"
        logger.info(f"rank: {stax.get_rank()}", log_for_all=True)
        # self.checkpointer = stax.Checkpointer(
        #     output_dir=path,
        #     max_to_keep=self.config.max_checkpoints_to_keep,
        #     # only allow train workers to write checkpoints
        #     active_processes=set(range(self.async_options.train_workers)),
        #     train_mesh=self.async_options.train_mesh,
        # )

        self.checkpointer = stax.OldCheckpointer(
            output_dir=path,
            max_to_keep=self.config.max_checkpoints_to_keep,
            # only allow train workers to write checkpoints
            train_mesh=self.async_options.train_mesh,
        )

    def make_save_tree(
        self,
        *,
        params: Optional[PyTree] = None,
        opt_state: Optional[PyTree] = None,
        metadata_metrics: Optional[dict[str, float]] = None,
    ):
        assert self.train_dataset is not None, "Train dataset must be set up to make save tree."
        assert self.train_mesh is not None, "train mesh should be intialized for checkpointing"

        dataset_state = {"train": self.train_dataset.save_checkpoint()}

        state = {
            "params": self.params if params is None else params,
            "opt_state": self.opt_state if opt_state is None else opt_state,
            "global_step": self.global_step,
            "dataset": dataset_state,
            "key": self.key.key,
        }

        metadata = {
            "writer_id": self.writer_id,
        }

        def convert_metric(x):
            return x.item() if isinstance(x, Array) else x

        if metadata_metrics is not None:
            metadata_metrics = jax.tree.map(convert_metric, metadata_metrics)
            metadata |= metadata_metrics

        return state, metadata

    def save_checkpoint(self, step: int):
        assert self.checkpointer is not None, "Checkpointer not set up."
        state, metadata = self.make_save_tree()
        logger.info(f"Saving checkpoint at step {step} ...", log_for_all=True)
        self.checkpointer.save(step=step, checkpoint_data=state, metadata=metadata)

    def restore_save_tree(self):
        assert self.checkpointer is not None, "Checkpointer not set up."
        assert self.model is not None, "Model not set up."
        assert self.tx is not None, "Optimizer not set up."
        assert self.train_dataset is not None, "Train dataset not set up."

        if self.checkpointer.latest_step is None:
            raise ValueError("No checkpoint found to restore from.")

        self.global_step = self.checkpointer.latest_step

        state, metadata = self.checkpointer.restore()
        self.params = jax.tree.map(lambda x, s: jax.device_put(x, s), state["params"], self.params_sharding)
        self.opt_state = jax.tree.map(lambda x, s: jax.device_put(x, s), state["opt_state"], self.opt_state_sharding)

        self.key.key = state["key"]
        train_state = state["dataset"].get("train")
        self.train_dataset.restore_checkpoint(train_state)

        self.writer_id = metadata.get("writer_id", None)

    def train_sync_weights(self):
        assert self.train_mesh is not None, "Train mesh must be set up to sync weights."

        with stax.Tracker(timer=True) as t:
            params_cast = jax.tree.map(
                lambda p: p.astype(self.config.loss_config.inference_config.params_dtype), self.params
            )

            params_cpu = jax.device_get(
                jax.device_put(params_cast, jax.sharding.NamedSharding(self.train_mesh, jax.P()))
            )

            if self.worker_rank == 0:
                address = self.transfer_server.address()
                for _ in range(self.async_options.inference_workers):
                    self.async_options.weight_sync_queue.put(address)

                while not self.async_options.weight_sync_queue.empty():
                    continue

                sharded_params = jax.device_put(params_cpu, jax.NamedSharding(self.local_mesh, jax.P()))
                for i in range(self.async_options.inference_workers):
                    uuid = self.weight_iteration * self.async_options.inference_workers + i
                    logger.info(f"[weight_sync] placing weights on uuid: {uuid}", log_for_all=True)
                    self.transfer_server.await_pull(uuid, {"params": sharded_params})

                for _ in range(self.async_options.inference_workers):
                    logger.info(
                        f"[weight_sync] {self.async_options.weight_sync_queue.get(timeout=TIMEOUT)}", log_for_all=True
                    )

            self.sync_train_workers(f"weight_sync_{self.weight_iteration}")

        self.weight_iteration += 1
        logger.info(f"[weight_sync] Weights sent to inference worker in {t.data['time']:.2f} seconds", log_for_all=True)
        return {"train/weight_sync_time": t.data["time"]}

    def _start_queue(self):
        def fn():
            assert self.train_dataset is not None, "Train dataset must be set up to fill queue."
            while True:
                if self.async_options.prompt_queue.empty():
                    for s in self.train_dataset(self.train_n_prompts):
                        self.async_options.prompt_queue.put(s)
                time.sleep(0.1)

        if self.worker_rank == 0:
            self.fill_thread = threading.Thread(target=fn, daemon=True)
            self.fill_thread.start()
        else:
            self.fill_thread = None

    def get_rollouts(self) -> tuple[list[InferenceRollout], dict]:
        rollouts = []
        weight_iterations = []
        num_filtered_rollouts = 0
        with stax.Tracker(timer=True) as t:
            while len(rollouts) < self.train_n_prompts_per_host:
                rollout = self.async_options.rollout_queue.get(timeout=TIMEOUT)
                if (lag_diff := (self.weight_iteration - rollout.weight_iteration)) <= self.config.async_config.max_lag:
                    rollouts.append(rollout)
                    weight_iterations.append(lag_diff)
                else:
                    num_filtered_rollouts += 1

        metrics = {
            "train/rollout_queue_wait_time": t.data["time"],
            "train/max_off_policy": max(weight_iterations),
            "train/min_off_policy": min(weight_iterations),
            "train/mean_off_policy": sum(weight_iterations) / len(weight_iterations),
            "train/num_filtered_rollouts": num_filtered_rollouts,
        }

        return rollouts, metrics

    def get_inference_metrics(self):
        metrics_per_worker = {i: [] for i in range(self.async_options.inference_workers)}
        while not self.async_options.inference_metrics_queue.empty():
            metrics = self.async_options.inference_metrics_queue.get(timeout=TIMEOUT)
            metrics_per_worker[metrics["worker_id"]].append(metrics)

        averaged_metrics = {
            f"inference/worker_{worker_id}/{k}": sum(d[k] for d in metrics_list) / len(metrics_list)
            if len(metrics_list) > 0
            else 0.0
            for worker_id, metrics_list in metrics_per_worker.items()
            for k in metrics_list[0].keys()
            if k != "worker_id"
        }

        metrics = averaged_metrics | {
            "inference/total_tps": sum(
                averaged_metrics[f"inference/worker_{worker_id}/decode_tps"] for worker_id in metrics_per_worker.keys()
            ),
        }

        return metrics

    def train_step(self, params, opt_state, local_batch, profile=False) -> tuple[PyTree, PyTree, dict]:
        assert self.train_fn is not None, "Train function not set up."
        assert self.shard_data_fn is not None, "Sharding function not set up."

        with stax.Tracker(timer=True) as t:
            global_train_batch = jax.tree.map(
                lambda x: rearrange(
                    x,
                    "(m g) ... -> g m ...",
                    m=self.config.data_config.batch_size // self.config.grad_accum_steps,
                    g=self.config.grad_accum_steps,
                ),  # [grad_accum_steps, minibatch_size, seq_len]
                self.shard_data_fn(local_batch),
            )

            if profile:
                trace_path = f"{self.gs_path}/{PROFILE}/step_{self.global_step}"
                out = stax.get_perf_func(
                    trace_path,
                    self.train_fn,  # type: ignore
                    params,
                    opt_state,
                    global_train_batch,
                )
            else:
                out = self.train_fn(params, opt_state, global_train_batch)

            # we have to sync weights so might as well block to get true step time
            jax.tree.map(lambda x: x.block_until_ready(), out)

        metrics = out["metrics"] | {"train/learner_step_time": t.data["time"]}
        return out["params"], out["opt_state"], metrics

    def train(self):
        assert self.params is not None and self.opt_state is not None, "Train state not initialized."
        assert self.shard_data_fn is not None, "Sharding function not set up."
        assert self.train_dataset is not None, "Train dataset not set up."
        assert self.writer is not None, "Writer not set up."
        assert self.checkpointer is not None, "Checkpointer not set up."

        logger.info("Precompiling test batch", log_for_all=True)

        # Profile first step
        # We overlap this with inference workers as they are async filling rollout queue
        # thus we have some time we can use to compile before we start the training loop
        local_test_batch = RLBatch.get_test_batch(
            batch_size=self.config.data_config.batch_size // self.config.async_config.train_workers,
            max_seq_len=self.config.loss_config.inference_config.max_seq_len,
        )
        self.train_step(self.params, self.opt_state, local_test_batch, profile=False)

        logger.info(f"Starting training loop at step {self.global_step}", log_for_all=True)
        while self.global_step < self.total_steps:
            with stax.Tracker(timer=True) as t:
                generations, local_rollout_metrics = self.get_rollouts()
                local_train_batch, local_train_batch_metrics = self.train_dataset.prepare_batch(generations, train=True)
                self.params, self.opt_state, train_metrics = self.train_step(
                    self.params, self.opt_state, local_train_batch
                )
                weight_sync_time = self.train_sync_weights()

            min_mem, max_mem = stax.get_memory()
            other_metrics = {
                "devices/memory_min": min_mem,
                "devices/memory_max": max_mem,
                "train/lr": self.opt_state[1].hyperparams["learning_rate"],
                "train/weight_sync_time": weight_sync_time["train/weight_sync_time"],
                "train/step_time": t.data["time"],
            }

            metrics = (
                train_metrics
                | metrics_all_reduce(local_train_batch_metrics, self.async_options.train_mesh)
                | metrics_all_reduce(local_rollout_metrics, self.async_options.train_mesh)
                | other_metrics
            )

            if stax.get_rank() == 0:
                metrics |= self.get_inference_metrics()

            generations_to_log = None
            if self.global_step % self.config.log_generations_every_n_steps == 0:
                generations_to_log = [(g.rollout_strs, g.sample.answer) for g in generations]

            self.writer(self.global_step, metrics, generations=generations_to_log)
            self.global_step += 1

            # save after you update state since if you want to save every 10 steps
            # you want to save after you have done 10 steps and resume at the 11th step
            if self.global_step % self.config.checkpoint_interval == 0 or self.global_step == self.total_steps:
                self.save_checkpoint(step=self.global_step)

        logger.info("Training complete.", log_for_all=True)

    def start(self):
        try:
            self.train()
        finally:
            self.finish()

    def sync_train_workers(self, name: str):
        sync_over_mesh(name, self.async_options.train_mesh)

    def block_checkpointer(self):
        assert self.checkpointer is not None, "Checkpointer not set up."
        self.checkpointer.block_until_ready()

    @partial(setup, component="cleanup")
    def finish(self):
        """Finalize training and clean up resources."""
        if self.checkpointer:
            logger.info("Cleaning up checkpointer resources...", log_for_all=True)
            self.block_checkpointer()

    @property
    def total_steps(self):
        return self.config.num_steps

    @property
    def resumed(self):
        assert self.checkpointer is not None, "Checkpointer not set up."
        return self.config.spot_training and self.checkpointer.latest_step is not None
