import os
from functools import partial

import jax
import optax
import stax
from omegaconf import DictConfig, OmegaConf
from stax import staxLogger as logger

from src.model import Model

from .config import TrainerConfig
from .steps import sft_step, standard_rl_step
from .utils import Key, set_jax_cache, setup


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

        # setup methods
        self._setup_jax()

        self._init_state()
        self._setup_model()
        self._setup_optimizer()
        self._setup_checkpointer()
        
        # self._setup_dataset()

        
        self._setup_functions()
        self._setup_train_state() 
        self._setup_writer()


        logger.info("Trainer initialization complete.")

    def validate_config(self):
        """Method to validate the TrainerConfig parameters."""
        # validate config here
        cfg = self.config
        if cfg.grad_accumulation < 1:
            raise ValueError("grad_accumulation must be at least 1")
        if cfg.num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if cfg.learning_rate_init < 0 or cfg.learning_rate_peak < 0 or cfg.learning_rate_end < 0:
            raise ValueError("learning rates must be non-negative")
        if cfg.optimizer not in ["adam", "adamw", "sgd"]:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
        if cfg.grad_clip is not None and cfg.grad_clip <= 0:
            raise ValueError("grad_clip must be positive if set")
        if cfg.best_metric is not None and cfg.checkpoint_interval % cfg.eval_interval != 0:
            raise ValueError("checkpoint_interval must be a multiple of eval_interval when best_metric is set")
        if (cfg.warmup_steps + cfg.decay_steps) > 1.0:
            raise ValueError("warmup_steps and decay_steps must sum to at most 1.0")
        if cfg.sharding_config.sharding_type not in ["single", "dp", "fsdp"]:
            raise ValueError("sharding_type must be one of 'single', 'dp', or 'fsdp'")

    @partial(setup, component="initialized state")
    def _init_state(self):
        self.train_fn = None
        self.eval_fn = None

        self.model = None
        self.tx = None

        self.params = None
        self.opt_state = None
        self.params_sharding = None
        self.opt_state_sharding = None

        self.train_dataset = None
        self.eval_dataset = None

        self.checkpointer = None
        self.best_checkpointer = None 


        self.writer_id = None  
        self.writer = None
        self.key = Key(self.config.seed)
        self.global_step = 0

    @partial(setup, component="metric logger")
    def _setup_writer(self):

        if writer_config := self.config.wandb_config:
            writer_kwargs =  dict() 
            if self.writer_id is not None:
                logger.info(f"Given existing wandb with id: {self.writer_id}")
                writer_kwargs['run_id'] = self.writer_id
            else: 
                logger.info("Starting new wandb run")
                writer_kwargs['config'] = OmegaConf.to_yaml(self.config)

            writer  = stax.WandBWriter(
                entity=os.environ.get("WANDB_ENTITY", ""),
                project=writer_config.project,
                **writer_kwargs
            )
        else:
            writer = stax.TextWriter()

        self.writer_id = writer.id
        self.writer = writer

    @partial(setup, component="JAX")
    def _setup_jax(self):
        """Setup JAX for distributed training on TPUs."""
        stax.init_distributed_jax()
        cache_path = self.config.gs_bucket + self.config.cache
        set_jax_cache(cache_path)

    @partial(setup, component="train/eval functions")
    def _setup_functions(self):
        """Setup training and evaluation functions."""
        assert self.tx is not None, "self.tx is None"
        assert self.model is not None, "self.model is None"

        params_shape, opt_state_shape= self.model.init_state(
            jax.random.PRNGKey(0), 
            tx=self.tx, 
            abstract=True
        )

        step_fn = sft_step
        self.train_fn, self.val_fn, self.shardings = stax.fn.get_steps_fn(
            step_fn, 
            self.model,
            self.tx, 
            has_aux=True,
            grad_steps=self.config.grad_steps,
            eval_steps=self.config.eval_steps,
            sharding=stax.ShardingConfig(
                params_shape=params_shape, 
                opt_state_shape=opt_state_shape,
                sharding_type=stax.ShardingType(self.config.sharding_config.sharding_type),
                opt_state_offload=self.config.sharding_config.opt_state_offload, 
                min_bytes_for_fsdp=self.config.sharding_config.min_bytes_for_fsdp,  
                data_shard_dim=self.config.sharding_config.data_shard_dim,
                weight_shard_dim=self.config.sharding_config.weight_shard_dim,
            )
        )


    @partial(setup, component="dataset")
    def _setup_dataset(self):
        """Setup the dataset for training."""
        pass

    @partial(setup, component="model")
    def _setup_model(self):
        """Setup the model for training."""
        self.model = Model(self.config.model_config)

    @partial(setup, component="init train state")
    def _setup_train_state(self):
        assert self.model is not None, "Model must be set up before train state init."
        assert self.tx is not None, "Optimizer must be set up before train state init."
        assert self.params_sharding is not None and self.opt_state_sharding is not None, "Sharding must be set up before train state init."

        sharding = (self.params_sharding, self.opt_state_sharding)
        out_state = self.model.init_state(rng=self.key(), tx=self.tx, sharding=sharding, abstract=False)
        self.params = out_state["params"]
        self.opt_state = out_state["opt_state"]

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

        path = f"{self.config.gs_bucket}/{self.config.checkpoint_gs_bucket}/"
        self.checkpointer  = stax.Checkpointer(
            output_dir=path, 
            max_to_keep=self.config.max_checkpoints_to_keep,
        )

        if self.has_best_ckpt:
            assert self.config.best_metric
            best_path = f"{path}/best/"
            self.best_checkpointer = stax.Checkpointer(
                output_dir=best_path,
                max_to_keep=1,
                best_key=self.config.best_metric.name,
                best_mode= "max" if self.config.best_metric.maximize else "min",
            )

    def make_save_tree(self):
        ...

    def restore_save_tree(self):
        ...

    def train(self):
        if self.train_fn is None or self.eval_fn is None:
            raise ValueError("Functions not initalized yet")

        for step in range(self.global_step, self.total_steps):

            # might have to do something here for RL 
            batch = self.train_dataset()
            out = self.train_fn(
                self.params,
                self.opt_state,
            )

            # val step 

            # val generations

            # checkpoint step 

    @property
    def has_best_ckpt(self):
        return self.config.best_metric is not None
    
    @property
    def total_steps(self):
        return self.config.num_steps
