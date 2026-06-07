from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import stax
from flax import linen as nn
from jax.sharding import Sharding, SingleDeviceSharding
from jaxtyping import Array, PyTree
from omegaconf import DictConfig
from optax import GradientTransformation
from stax import HFModelBase
from stax import staxLogger as logger

from .config import ModelConfig
from .qwen3 import KVCache, Qwen3
from .utils import convert_dtype, get_qwen_3_weights, save_to_hf

sizes = [0.6, 1.7, 4, 8]
model_names = [f"Qwen/Qwen3-{size}B" for size in sizes] + [f"Qwen/Qwen3-{size}B-Base" for size in sizes]
shardingType = Optional[PyTree[Sharding]]


class Model(HFModelBase):
    def __init__(self, config: DictConfig | ModelConfig):
        self.config = config
        self.validate_config()
        is_base = self.config.hf_model_name.endswith("Base")
        self.model = Qwen3.from_config(config.qwen_config, is_base=is_base)

    def validate_config(self):
        if self.config.hf_model_name not in model_names:
            raise ValueError(f"Expected model size to be in {model_names}, got {self.config.hf_model_name}")

    def init_state(
        self, rng: Array, tx: Optional[GradientTransformation], *, sharding: shardingType = None, abstract: bool = False
    ) -> PyTree:
        x_init = jnp.ones((1, self.config.qwen_config.sequence_len), dtype=jnp.int32)
        seq_lens = jnp.array([1])

        def init_state(rng, x_init, sequence_lens):
            params = jax.eval_shape(self.model.init, rngs=rng, x=x_init, sequence_lens=sequence_lens, kv_cache=None)[
                "params"
            ]
            out_state = {"params": params}
            if tx:
                out_state["opt_state"] = tx.init(params)
            return out_state

        if abstract:
            return jax.eval_shape(init_state, rng, x_init, seq_lens)

        out_state = init_state(rng, x_init, seq_lens)
        out_state["params"] = self.load_from_hf(out_state["params"], self.config.hf_model_name)

        if sharding is None:
            single_sharding = SingleDeviceSharding(jax.local_devices()[0])
            sharding = jax.tree.map(lambda _: single_sharding, out_state)

        if out_state.keys() != sharding.keys():
            raise ValueError(f"sharding keys do not match got {sharding.keys()} expected {out_state.keys()}")
        out_state = jax.tree.map(lambda x, s: jax.device_put(x, s), out_state, sharding)

        table = nn.tabulate(self.model, rngs=jax.random.PRNGKey(0), depth=1)
        logger.info(table(x=x_init, sequence_lens=seq_lens, kv_cache=None))

        return out_state

    def load_from_hf(self, params: PyTree, model_name: str) -> PyTree:
        return get_qwen_3_weights(params, name=model_name)

    def init_kv_cache(self, batch_size: int, length: int, sharding: KVCache, dtype: str = "bfloat16") -> list[KVCache]:
        if length > self.config.qwen_config.sequence_len + 1024:
            raise ValueError(
                f"Requested KV cache length {length} exceeds maximum of {self.config.qwen_config.sequence_len + 1024}"
            )

        @partial(jax.jit, out_shardings=sharding)
        def _init():
            def zeros():
                return jnp.zeros(
                    (
                        batch_size,
                        length,
                        self.config.qwen_config.n_groups,
                        self.config.qwen_config.head_dim,
                    ),
                    dtype=convert_dtype(dtype),
                )

            return KVCache(k=zeros(), v=zeros(), length=0)

        return [_init() for _ in range(self.config.qwen_config.n_layers)]

    def load_from_ckpt(
        self, path: str, step_number: Optional[int] = None, use_best=False
    ) -> Tuple[int, PyTree, dict[str, float]]:
        assert (step_number is not None) ^ use_best, "Either step_number or use_best must be set."
        path = f"{path}/checkpoints/"
        if use_best:
            path += "best/"

        checkpointer = ocp.CheckpointManager(directory=path, options=ocp.CheckpointManagerOptions())

        if step_number == -1:
            step_number = None

        if step_number is None:
            step_number = checkpointer.latest_step()
            if step_number is None:
                raise ValueError("No checkpoints found.")

        save_tree = self.init_state(jax.random.PRNGKey(0), tx=None, abstract=True)
        # use np.ndarray to load on CPU from sharded arrays (https://github.com/google/orbax/issues/648)
        restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), save_tree)
        restored = checkpointer.restore(
            step=step_number,
            args=ocp.args.Composite(
                state=ocp.args.PyTreeRestore(save_tree, restore_args=restore_args, partial_restore=True),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        assert hasattr(restored, "state"), "Restored object has no attribute 'state'"

        return step_number, restored.state["params"], restored.metadata  # type: ignore

    def load_from_ckpt_old(
        self,
        path: str,
        step_number: Optional[int] = None,
    ) -> Tuple[int, PyTree, dict[str, float]]:
        path = f"{path}/checkpoints/"

        checkpointer = stax.OldCheckpointer(path)
        if step_number == -1:
            step_number = None

        if step_number is None:
            step_number = checkpointer.latest_step
            if step_number is None:
                raise ValueError("No checkpoints found.")

        state, metadata = checkpointer.restore(step=step_number)
        return step_number, state["params"], metadata

    def save_hf(self, path: str, params: PyTree) -> None:
        save_to_hf(path, params, self.config.hf_model_name)

    def __call__(
        self,
        params: PyTree,
        *,
        x: Array,
        sequence_lens: Array,
        kv_cache: Optional[list[KVCache]] = None,
    ) -> tuple[Array, list[KVCache]]:
        logits, cache = self.model.apply(params, x, sequence_lens, kv_cache)

        return logits, cache

    def apply(
        self,
        params: PyTree,
        *,
        x: Array,
        sequence_lens: Array,
        kv_cache: Optional[list[KVCache]] = None,
    ) -> tuple[Array, list[KVCache]]:
        """
        Forward pass of the model
        Args:
            params: Model parameters.
            x: Input tokens of shape (B, T).
            sequence_lens: Sequence lengths of shape (B,).
            kv_cache: Optional list of KVCache for each layer.
        Returns:
            logits: Output logits of shape (B, T, vocab_size).
            out_cache: Optional list of KVCache for each layer if kv_cache was provided.
        """
        return self(params, x=x, sequence_lens=sequence_lens, kv_cache=kv_cache)

    @property
    def sequence_len(self) -> int:
        """Max sequence length supported by the model"""
        return self.config.qwen_config.sequence_len

    @property
    def activation_dtype(self):
        return convert_dtype(self.config.qwen_config.activation_dtype)
