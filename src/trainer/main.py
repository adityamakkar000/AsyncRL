import json
import os
from functools import partial
from typing import Dict, Optional

import jax
import optax
import stax
from dotenv import load_dotenv
from jax.experimental.multihost_utils import sync_global_devices
from jaxtyping import PyTree
from omegaconf import DictConfig, OmegaConf
from stax import TrainFn
from stax import staxLogger as logger

from src.constants import CACHE, CHECKPOINTS, GS_BUCKET
from src.data import RLBatch
from src.inference_engine import InferenceEngine
from src.model import Model

from .config import TrainerConfig
from .loss import compute_aux_metrics, get_rl_step_fn
from .utils import Key, set_jax_cache, setup, write_to_gcs

load_dotenv()


class Trainer:
    """
    A Trainer class to handle the training process of a machine learning model using JAX on TPUs.
    """

    def __init__(self, config: TrainerConfig | DictConfig):
        """
        Initialize the training module.

        Args:
            config (TrainerConfig): Configuration object containing model, data, training parameters, etc.
        """

        self.config = config
        self.validate_config()
        self._setup_jax()

        logger.info(f"Starting training for {self.config.experiment_name}")

        with stax.Tracker(timer=True) as tracker:
            self._init_state()
            self._setup_model()
            self._setup_optimizer()
            self._setup_checkpointer()
            self._setup_functions()
            self._setup_dataset()
            self._setup_train_state()
            self._setup_writer()
            self._setup_inference_engine()

            assert self.checkpointer is not None, "Checkpointer not set up."
            if self.checkpointer.latest_step is None:
                logger.info("Saving intial checkpoint ...")
                self.save_checkpoint(step=self.global_step)
                dict_config = json.dumps(OmegaConf.to_container(self.config))
                config_path = f"{GS_BUCKET}/{self.config.experiment_name}/config.json"
                write_to_gcs(config_path, dict_config)
                self.checkpointer.wait_until_finished()

            sync_global_devices("Trainer initialization")

        logger.info("Training Configuration:\n" + OmegaConf.to_yaml(config))
        logger.info(f"Trainer initialization complete in {tracker.data['time']:.2f} seconds")

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
        if cfg.best_metric is not None and cfg.checkpoint_interval % cfg.val_interval != 0:
            raise ValueError("checkpoint_interval must be a multiple of val_interval when best_metric is set")
        if (cfg.warmup_steps + cfg.decay_steps) > 1.0:
            raise ValueError("warmup_steps and decay_steps must sum to at most 1.0")
        if cfg.sharding_config.sharding_type not in ["single", "dp", "fsdp"]:
            raise ValueError("sharding_type must be one of 'single', 'dp', or 'fsdp'")
        if cfg.loss_config.rl_config.algorithm not in ["grpo", "dr_grpo", "dapo"]:
            raise ValueError(f"Unsupported RL algorithm: {cfg.loss_config.rl_config.algorithm}")

        # TODO: check if group size divices batch size

    @partial(setup, component="initialized state")
    def _init_state(self):
        self.train_fn = None

        self.model = None
        self.tx = None
        self.inference_engine = None

        self.params = None
        self.opt_state = None
        self.params_sharding = None
        self.opt_state_sharding = None

        self.train_dataset = None
        self.val_dataset = None

        self.checkpointer = None
        self.best_checkpointer = None

        self.writer_id = None
        self.writer = None
        self.key = Key(self.config.seed)
        self.global_step = 0

    @partial(setup, component="metric logger")
    def _setup_writer(self):
        if writer_config := self.config.wandb_config:
            writer_kwargs = dict()
            if self.writer_id is not None:
                writer_kwargs["run_id"] = self.writer_id
            else:
                writer_kwargs["config"] = OmegaConf.to_object(self.config)

            writer = stax.WandBWriter(
                entity=os.getenv("WANDB_ENTITY", ""), project=writer_config.project, **writer_kwargs
            )
        else:
            writer = stax.TextWriter()

        self.writer_id = writer.id
        self.writer = writer

    @partial(setup, component="JAX")
    def _setup_jax(self):
        """Setup JAX for distributed training on TPUs."""
        stax.init_distributed_jax()
        cache_path = f"{GS_BUCKET}/{CACHE}"
        set_jax_cache(cache_path)

    @partial(setup, component="train and val functions")
    def _setup_functions(self):
        """Setup training and validation functions."""
        assert self.tx is not None, "self.tx is None"
        assert self.model is not None, "self.model is None"

        abstract_state = self.model.init_state(jax.random.PRNGKey(0), tx=self.tx, abstract=True)
        params_shape, opt_state_shape = abstract_state["params"], abstract_state["opt_state"]

        step_fn = get_rl_step_fn(self.config.loss_config.rl_config)

        # val fn not needed since we just care about val reward, not loss
        train_fn, _val_fn, shardings = stax.fn.get_steps_fn(
            step_fn,
            self.model,
            self.tx,
            has_aux=True,
            grad_steps=self.config.grad_accum_steps,
            val_steps=self.config.val_steps,
            sharding=stax.ShardingConfig(
                params_shape=params_shape,
                opt_state_shape=opt_state_shape,
                sharding_type=stax.ShardingType(self.config.sharding_config.sharding_type),
                opt_state_offload=self.config.sharding_config.opt_state_offload,
                min_bytes_for_fsdp=self.config.sharding_config.min_bytes_for_fsdp,
                data_shard_dim=self.config.sharding_config.data_shard_dim,
                weight_shard_dim=self.config.sharding_config.weight_shard_dim,
            ),
        )

        self.params_sharding, self.opt_state_sharding = shardings.param_sharding, shardings.opt_state_sharding
        self.train_fn = train_fn

        def train_step(param: PyTree, opt_state: PyTree, batch: RLBatch) -> Dict[str, PyTree]:
            aux_metrics = {}
            for step in range(self.config.loss_config.grad_steps):
                out = train_fn(self.params, self.opt_state, batch)
                self.params = out["params"]
                self.opt_state = out["opt_state"]
                aux_metrics |= {f"{k}_step_{step}": v for k, v in out["aux_metrics"].items()}

            return {
                "params": self.params,
                "opt_state": self.opt_state,
                "aux_metrics": aux_metrics | compute_aux_metrics(batch),
            }

        self.train_step: TrainFn = train_step

    @partial(setup, component="dataset")
    def _setup_dataset(self):
        """Setup the dataset for training."""
        # TODO: (chinmay)
        # setup the train and the val dataset
        self.train_dataset = None
        self.val_dataset = None

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

        if self.config.spot_training and self.checkpointer.latest_step is not None:
            logger.info("Spot training enabled and checkpoint found, skipping parameter initialization.")
            self.restore_save_tree()
            return

        logger.info("Initializing new run ...")
        sharding = {"params": self.params_sharding, "opt_state": self.opt_state_sharding}
        out_state = self.model.init_state(rng=self.key(), tx=self.tx, sharding=sharding, abstract=False)
        self.params = out_state["params"]
        self.opt_state = out_state["opt_state"]

        logger.info(f"Params intialized with total size: {self.model.count_params(self.params):_} parameters.")

    @partial(setup, component="inference engine")
    def _setup_inference_engine(self):
        """Setup the inference engine for evaluation and generation rollouts."""
        if self.config.loss_config.inference_config is None:
            logger.info("No inference config provided, skipping inference engine setup.")
            return

        assert self.model is not None, "Model must be set up before inference engine init."
        assert self.params is not None, "Train state must be initialized before inference engine init."

        inference_params = {"params": self.params}
        # inference_engine requires {params: params...}
        self.inference_engine = InferenceEngine(
            model=self.model, params=inference_params, config=self.config.loss_config.inference_config
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

        optimizer_args = {
            "learning_rate": lr_scheduler,
        }
        if self.config.weight_decay is not None and self.config.optimizer == "adamw":
            optimizer_args["weight_decay"] = self.config.weight_decay
        self.tx = optax.chain(
            clip,
            optax.inject_hyperparams(optimizer)(
                **optimizer_args,
            ),
        )

    @partial(setup, component="checkpointer")
    def _setup_checkpointer(self):
        """Setup checkpointing mechanism."""

        path = f"{GS_BUCKET}/{self.config.experiment_name}/{CHECKPOINTS}/"
        self.checkpointer = stax.Checkpointer(
            output_dir=path,
            max_to_keep=self.config.max_checkpoints_to_keep,
            best_key=self.config.best_metric.name if self.has_best_ckpt else None,  # type: ignore
            best_mode="max" if self.has_best_ckpt and self.config.best_metric.maximize else "min",  # type: ignore
        )

    def make_save_tree(
        self,
        step: int,
        *,
        params: Optional[PyTree] = None,
        opt_state: Optional[PyTree] = None,
        metadata_metrics: Optional[dict[str, float]] = None,
    ):
        # TODO: (chinmay) save dataset state
        # dataset_state = {"train": self.train_dataset.save_checkpoint(), "val": self.val_dataset.save_checkpoint()}

        state = {
            "params": params if params else self.params,
            "opt_state": opt_state if opt_state else self.opt_state,
            "global_step": self.global_step,
            "dataset": None,  # TODO: (chinmay) add dataset state here
            "key": jax.device_get(self.key.key),
        }
        metadata = {
            "writer_id": self.writer_id,
        }
        if metadata_metrics is not None:
            metadata |= metadata_metrics

        return state, metadata

    def save_checkpoint(self, step: int, metadata_metrics: Optional[dict[str, float]] = None):
        assert self.checkpointer is not None, "Checkpointer not set up."
        state, metadata = self.make_save_tree(step, metadata_metrics=metadata_metrics)
        logger.info(f"Saving checkpoint at step {step} ...")
        self.checkpointer.save_checkpoint(step=step, save_tree=state, metadata=metadata)

    def restore_save_tree(self, use_best: bool = False):
        assert self.checkpointer is not None, "Checkpointer not set up."
        assert self.model is not None, "Model not set up."
        assert self.tx is not None, "Optimizer not set up."
        if use_best and not self.has_best_ckpt:
            use_best = False
            logger.warning(
                "Best checkpoint requested but best_metric not set in config. Loading latest checkpoint instead."
            )

        if self.checkpointer.latest_step is None:
            raise ValueError("No checkpoint found to restore from.")

        self.global_step = self.checkpointer.latest_step

        shardings = {
            "params": self.params_sharding,
            "opt_state": self.opt_state_sharding,
        }
        out = self.model.init_state(rng=jax.random.PRNGKey(0), tx=self.tx, sharding=shardings, abstract=True)

        # don't need metadata
        save_tree, _ = self.make_save_tree(
            step=-1,
            params=out["params"],
            opt_state=out["opt_state"],
        )

        restored_ckpt = self.checkpointer.restore(state=save_tree, use_best=use_best)

        state, metadata = restored_ckpt["state"], restored_ckpt["metadata"]

        self.params = state["params"]
        self.opt_state = state["opt_state"]
        self.key.key = state["key"]

        self.writer_id = metadata.get("writer_id", None)

        # TODO: (chinmay) restore dataset state
        # self.train_dataset.load_from_state(state["dataset"]["train"])
        # self.val_dataset.load_from_state(state["dataset"]["val"])

    def train(self):
        self._setup_inference_engine()
        assert self.train_fn is not None, "Train function not set up."
        assert self.inference_engine is not None, "Inference engine not set up."
        # assert self.train_dataset is not None, "Train dataset not set up."
        assert self.writer is not None, "Writer not set up."
        assert self.checkpointer is not None, "Checkpointer not set up."

        logger.info("Starting training loop...")
        while self.global_step < self.total_steps:
            # get train batch
            # TODO: (chinmay)
            # prompts = self.train_dataset()

            # get rollouts
            _output = self.inference_engine(
                ["what is your name"], jax.random.PRNGKey(0), {"params": self.params}, detokenize=True
            )

            # prepare batch
            # TODO: (chinmay)
            # train_batch = self.train_dataset.prepare_batch(rollouts)
            breakpoint()
            train_batch = ...

            out = self.train_step(self.params, self.opt_state, train_batch)
            self.params = out["params"]
            self.opt_state = out["opt_state"]
            self.writer(self.global_step, out["aux_metrics"])

            if self.global_step % self.config.val_interval == 0:
                # TODO: val step here
                ...

            self.global_step += 1

        logger.info("Training complete.")

    @partial(setup, component="cleanup")
    def finish(self):
        """Finalize training and clean up resources."""
        if self.writer:
            logger.info("Cleaning up writer resources...")
            self.writer.finish()
        if self.checkpointer:
            logger.info("Cleaning up checkpointer resources...")
            self.checkpointer.wait_until_finished()

    @property
    def has_best_ckpt(self):
        return self.config.best_metric is not None

    @property
    def total_steps(self):
        return self.config.num_steps
