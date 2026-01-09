from typing import Optional

import jax
import jax.numpy as jnp
from jax.sharding import Sharding, SingleDeviceSharding
from jaxtyping import Array, PyTree
from omegaconf import DictConfig
from optax import GradientTransformation
from stax.model_module import HFModelBase

from src.model.qwen3 import KVCache, Qwen3

from .config import ModelConfig
from .utils import get_qwen_3_weights

sizes = [0.6, 1.7, 4, 8]
model_names = [f"Qwen/Qwen3-{size}B" for size in sizes]
shardingType = Optional[PyTree[Sharding]]


class Model(HFModelBase):
    def __init__(self, config: DictConfig[ModelConfig]):
        self.config = config
        self.validate_config()
        self.model = Qwen3.from_config(config.model_config)

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
        else:
            out_state = init_state(rng, x_init, seq_lens)
            out_state["params"] = get_qwen_3_weights(out_state["params"], name=self.config.hf_model_name)

        if sharding is None:
            single_sharding = SingleDeviceSharding(jax.devices()[0])
            sharding = jax.tree.map(lambda _: single_sharding, out_state)

        if out_state.keys() != sharding.keys():
            raise ValueError(f"sharding keys do not match got {sharding.keys()} expected {out_state.keys()}")
        out_state = jax.tree.map(lambda x, s: jax.device_put(x, s), out_state, sharding)

        return out_state

    def save_to_hf():
        # TODO: implement method to load model weights to huggingface
        return

    def init_kv_cache(self, x: Array) -> list[KVCache]:
        B = x.shape[0]
        n_layers = self.config.model_config.n_layers
        n_groups = self.config.model_config.n_groups
        max_sequence_len = self.config.model_config.sequence_len
        head_dim = self.config.model_config.head_dim

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
