from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import Sharding, SingleDeviceSharding
from jaxtyping import Array, PyTree
from omegaconf import DictConfig
from optax import GradientTransformation
from stax.model_module import HFModelBase

from src.model.qwen3 import KVCache, Qwen3

from .config import ModelConfig
from .utils import get_qwen_3_weights, save_to_hf

sizes = [0.6, 1.7, 4, 8]
model_names = [f"Qwen/Qwen3-{size}B" for size in sizes]
shardingType = Optional[PyTree[Sharding]]


class Model(HFModelBase):
    def __init__(self, config: DictConfig[ModelConfig]):
        self.config = config
        self.validate_config()
        self.model = Qwen3.from_config(config.qwen_config)

    def validate_config(self):
        if self.config.hf_model_name not in model_names:
            raise ValueError(f"Expected model size to be in {model_names}, got {self.config.hf_model_name}")

    def init_state(
        self, rng: Array, tx: Optional[GradientTransformation], *, sharding: shardingType = None, abstract: bool = False
    ) -> PyTree:
        x_init = jnp.ones((1, 1), dtype=jnp.int32)
        seq_lens = jnp.array([1])

        @jax.jit
        def init_state(rng, x_init, seq_lens):
            params = self.model.init(rngs=rng, x=x_init, seq_lens=seq_lens, kv_cache=None)["params"]
            out_state = {"params": params}
            if tx:
                out_state["opt_state"] = tx.init(params)
            return out_state

        if abstract:
            return jax.eval_shape(init_state, rng, x_init, seq_lens)

        out_state = init_state(rng, x_init, seq_lens)
        out_state["params"] = self.load_from_hf(out_state["params"], self.config.hf_model_name)

        if sharding is None:
            single_sharding = SingleDeviceSharding(jax.devices()[0])
            sharding = jax.tree.map(lambda _: single_sharding, out_state)

        if out_state.keys() != sharding.keys():
            raise ValueError(f"sharding keys do not match got {sharding.keys()} expected {out_state.keys()}")
        out_state = jax.tree.map(lambda x, s: jax.device_put(x, s), out_state, sharding)

        return out_state

    def load_from_hf(self, params: PyTree, model_name: str) -> PyTree:
        return get_qwen_3_weights(params, name=model_name)

    def init_kv_cache(self, x: Array) -> list[KVCache]:
        B = x.shape[0]
        n_layers = self.config.qwen_config.n_layers
        n_groups = self.config.qwen_config.n_groups
        max_sequence_len = self.config.qwen_config.sequence_len
        head_dim = self.config.qwen_config.head_dim

        initial_cache: list[KVCache] = []
        for _ in range(n_layers):
            length = 0
            k = jnp.zeros((B, max_sequence_len, n_groups, head_dim), dtype=jnp.bfloat16)
            v = jnp.zeros((B, max_sequence_len, n_groups, head_dim), dtype=jnp.bfloat16)
            _cache = KVCache(
                k=k,
                v=v,
                length=length,
            )
            initial_cache.append(_cache)

        return initial_cache

    def load_from_ckpt(self, path: str, step_number: Optional[int] = None, use_best=False):
        assert (step_number is not None) ^ use_best, "Either step_number or use_best must be set."
        path = f"{path}/checkpoints/"
        if use_best:
            path += "best/"

        checkpointer = ocp.CheckpointManager(directory=path, options=ocp.CheckpointManagerOptions())

        if step_number == -1:
            step_number = None

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
        assert restored.state, "Restored state has no 'params' key"

        return restored.state["params"]

    def save_hf(self, path: str, params: PyTree) -> None:
        """Saves the model parameters in a local safetensors file. Inverse of load_from_hf."""
        save_to_hf(path, params, self.config.hf_model_name)

    def __call__(
        self,
        params: PyTree,
        *,
        x: Array,
        sequence_lens: Array,
        kv_cache: Optional[list[KVCache]] = None,
    ) -> PyTree:
        logits, cache = self.model.apply(params, x, sequence_lens, kv_cache)

        return logits, cache

    def apply(
        self,
        params: PyTree,
        *,
        x: Array,
        sequence_lens: Array,
        kv_cache: Optional[list[KVCache]] = None,
    ) -> PyTree:
        return self(params, x=x, sequence_lens=sequence_lens, kv_cache=kv_cache)
