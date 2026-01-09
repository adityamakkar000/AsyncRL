from functools import partial

import optax
import stax

from src.model import Model

from .config import TrainerConfig
from .utils import Key, setup


class Trainer:
    """
    A Trainer class to handle the training process of a machine learning model using JAX on TPUs.
    """

    def __init__(self, config: TrainerConfig):
        """
        Initialize the training module.

        Args:
            config (TrainerConfig): Configuration object containing model, data, training parameters, etc.
        """
        
        self.config = config
        self.validate_config()

        # setup methods 

        self._init_state()

        self._setup_jax()
        self._setup_dataset()
        self._setup_optimizer()
        self._setup_model()
        self._checkpointer_setup()
    

    def validate_config(self):
        """ Method to validate the TrainerConfig parameters. """
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
    
    def _init_state(self):

        self.model = None 
        self.tx = None

        self.params = None 
        self.opt_state = None
        self.sharding = None

        self.train_dataset = None
        self.eval_dataset = None

        self.key = Key(self.config.seed)

    @partial(setup, component="JAX")
    def _setup_jax(self):
        """ Setup JAX for distributed training on TPUs. """
        stax.init_distributed_jax()

    @partial(setup, component="dataset")
    def _setup_dataset(self):
        """ Setup the dataset for training. """
        pass
    
    @partial(setup, component="model")
    def _setup_model(self):
        """ Setup the model for training. """
        if self.tx is None:
            raise ValueError("Optimizer must be set up before the model.")

        self.model = Model(self.config.model_config)
        out_state = self.model.init_state(
            rng=self.key(),
            tx=self.tx,
            sharding=self.sharding, 
            abstract=False
        )
        self.params = out_state['params']
        self.opt_state = out_state['opt_state']

    @partial(setup, component="optimizer")
    def _setup_optimizer(self):
        """ Setup the optimizer and learning rate scheduler. """
        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=self.config.learning_rate_init,
            peak_value=self.config.learning_rate_peak,
            warmup_steps=int(self.config.warmup_steps * self.config.num_steps),
            decay_steps=int(self.config.decay_steps * self.config.num_steps),
            end_value=self.config.learning_rate_end,
        ) 

        match self.config.optimizer:
            case "adam":
                optimizer= optax.adam
            case "adamw":
                optimizer = optax.adamw
            case "sgd":
                optimizer = optax.sgd
            case _:
                raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        clip = optax.clip_by_global_norm(self.config.grad_clip) if self.config.grad_clip is not None else optax.identity()

        optimizer_args ={
            'learning_rate': lr_scheduler,
        }
        if self.config.weight_decay is not None and self.config.optimizer == "adamw":
            optimizer_args['weight_decay'] = self.config.weight_decay
        self.tx = optax.chain(
            clip,
            optax.inject_hyperparams(optimizer)(
                **optimizer_args,
            )
        )

    @partial(setup, component="checkpointer")
    def _checkpointer_setup(self):
        """ Setup checkpointing mechanism. """
        pass



