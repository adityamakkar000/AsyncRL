import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import stax
from flax import linen as nn
from hydra.utils import instantiate
from jax.sharding import Sharding
from jaxtyping import Array, PyTree
from omegaconf import DictConfig
from optax import GradientTransformation
from stax import HFModelBase
from stax import staxLogger as logger

from .config import BaseModel, Int8Array, KVCache, ModelConfig
from .models.kernels import fused_linear_selection
from .utils import convert_dtype, get_lm_head_weights, mesh_size

shardingType = PyTree[Sharding] | None


class Model(HFModelBase):
    def __init__(self, config: DictConfig | ModelConfig):
        self.config = config
        self.model: BaseModel = instantiate(config.model_args)
        assert isinstance(self.model, BaseModel), (
            f"Expected model to be an instance of BaseModel, got {type(self.model)}"
        )

    def init_state(
        self, rng: Array, tx: GradientTransformation | None, *, sharding: stax.Shardings | None = None
    ) -> PyTree:
        def init_fn(rng, x, sequence_lens):
            params = self.model.init(rngs=rng, x=x, sequence_lens=sequence_lens, kv_cache=None)["params"]
            return {"params": params, "opt_state": tx.init(params) if tx else None}

        batch = mesh_size()
        abstract_x = jax.ShapeDtypeStruct((batch, self.sequence_len), jnp.int32)
        abstract_seq_lens = jax.ShapeDtypeStruct((batch,), jnp.int32)

        if sharding is None:
            logger.info("No sharding is given returning abstract state")
            return jax.eval_shape(init_fn, rng, abstract_x, abstract_seq_lens)

        local_batch = jax.local_device_count()
        x_init = np.ones((local_batch, self.sequence_len), dtype=np.int32)
        seq_lens = np.ones((local_batch,), dtype=np.int32)

        sharded_init_args = sharding.shard_data((rng, x_init, seq_lens))
        out_shardings = {"params": sharding.param_sharding, "opt_state": sharding.opt_state_sharding if tx else None}

        out_state = jax.jit(init_fn, out_shardings=out_shardings)(*sharded_init_args)

        logger.info(jax.tree.map(lambda x: (x.shape, x.sharding.spec), out_state["params"]))
        out_state["params"] = self.load_from_hf(out_state["params"], self.config.hf_model_name)

        table = nn.tabulate(self.model, rngs=jax.random.PRNGKey(0), depth=1)
        logger.info(
            table(x=np.ones(abstract_x.shape, np.int32), sequence_lens=np.ones((batch,), np.int32), kv_cache=None)
        )
        return out_state

    def init_kv_cache(self, batch_size: int, length: int, sharding: KVCache, dtype: str = "bfloat16") -> KVCache:
        if length > self.sequence_len:
            raise ValueError(f"Requested KV cache length {length} exceeds maximum of {self.sequence_len + 1024}")

        n_layers, n_kv_heads, head_dim = self.model.kv_shape
        cache_dtype = convert_dtype(dtype)

        @jax.jit
        def _init():
            def zeros(out_sharding, dtype=cache_dtype, shape=(n_layers, batch_size, n_kv_heads, length, head_dim)):
                return jnp.zeros(shape, dtype=dtype, out_sharding=out_sharding)

            def cache(out_sharding):
                if cache_dtype == jnp.int8:
                    return Int8Array(
                        value=zeros(out_sharding),
                        scale=zeros(out_sharding, dtype=jnp.bfloat16, shape=(n_layers, batch_size, n_kv_heads, length)),
                    )
                return zeros(out_sharding)

            return KVCache(
                k=cache(sharding.k),
                v=cache(sharding.v),
                length=jnp.zeros((batch_size,), dtype=jnp.int32, out_sharding=sharding.length),  # type: ignore
            )

        return _init()

    def load_from_ckpt(
        self, path: str, step_number: int | None = None, use_best=False
    ) -> tuple[int, PyTree, dict[str, float]]:
        assert not (use_best and step_number is not None), "Set either step_number or use_best, not both."
        path = f"{path}/checkpoints/"
        if use_best:
            path += "best/"

        checkpointer = ocp.CheckpointManager(directory=path, options=ocp.CheckpointManagerOptions(read_only=True))

        if step_number == -1:
            step_number = None

        if step_number is None:
            step_number = checkpointer.latest_step()
            if step_number is None:
                raise ValueError("No checkpoints found.")

        save_tree = {"params": self.init_state(jax.random.PRNGKey(0), tx=None, sharding=None)["params"]}
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

    def load_from_hf(self, params: PyTree, model_name: str) -> PyTree:
        return self.model.load_from_hf(params, model_name)

    def save_hf(self, path: str, params: PyTree) -> None:
        self.model.save_to_hf(path, params, self.config.hf_model_name)

    def __call__(
        self,
        params: PyTree,
        *,
        x: Array,
        sequence_lens: Array,
        kv_cache: list[KVCache] | None = None,
        apply_lm_head: bool = True,
    ) -> tuple[Array, list[KVCache]]:
        logits, cache = self.model.apply(params, x, sequence_lens, kv_cache, apply_lm_head)

        return logits, cache  # type: ignore

    def get_logprobs(self, params: PyTree, tokens: Array, seq_lens: Array) -> Array:
        """
        Log probability of each target token under the model.
        Args:
            params: Model parameters.
            tokens: Input tokens of shape (B, T).
            seq_lens: Sequence lengths of shape (B,).
        Returns:
            x_logprobs: Log probabilities of shape (B, T - 1).
        """

        if self.config.fused_chunk_size is not None:
            logger.info(f"Using fused logprob path (chunk_size={self.config.fused_chunk_size})")
            x_logprobs, _ = self.fused_selection_call(
                {"params": params},
                x=tokens,
                sequence_lens=seq_lens,
                tokens=tokens,
                kv_cache=None,
                chunk_size=self.config.fused_chunk_size,
            )
        else:
            target_tokens = tokens[:, 1:]  # B, T-1
            x_logits, _ = self.apply(
                {"params": params},
                x=tokens,
                sequence_lens=seq_lens,
                kv_cache=None,
            )

            x_logprobs = jax.nn.log_softmax(x_logits[:, :-1, :], axis=-1)
            x_logprobs = jnp.take_along_axis(x_logprobs, target_tokens[..., None], axis=-1)[..., 0]

        return x_logprobs

    def fused_selection_call(
        self,
        params: PyTree,
        *,
        x: Array,
        sequence_lens: Array,
        tokens: Array,
        kv_cache: list[KVCache] | None = None,
        chunk_size: int = 1024,
    ) -> tuple[Array, list[KVCache]]:
        """
        Forward pass of the model
        Args:
            params: Model parameters.
            x: Input tokens of shape (B, T).
            sequence_lens: Sequence lengths of shape (B,).
            kv_cache: Optional list of KVCache for each layer.
        Returns:
            logits: Output logits of shape (B, T, D) and then selected and returns (B, T)
            out_cache: Optional list of KVCache for each layer if kv_cache was provided.
        """
        B, T = x.shape
        hidden_output, cache = self.model.apply(
            params, x=x, sequence_lens=sequence_lens, kv_cache=kv_cache, apply_lm_head=False
        )  # B, T, D
        D = hidden_output.shape[-1]
        weights = get_lm_head_weights(params["params"], self.model.tie_weights)  # D, V

        flat_hidden = hidden_output.reshape(B * T, D)
        flat_targets = jnp.roll(tokens, -1, axis=-1).reshape(B * T)
        logprobs = fused_linear_selection(flat_hidden, weights, flat_targets, chunk_size).reshape(B, -1)  # B, T

        return logprobs[:, :-1], cache  # type: ignore

    def apply(
        self,
        params: PyTree,
        *,
        x: Array,
        sequence_lens: Array,
        kv_cache: list[KVCache] | None = None,
        apply_lm_head: bool = True,
    ) -> tuple[Array, list[KVCache]]:
        """
        Forward pass of the model
        Args:
            params: Model parameters.
            x: Input tokens of shape (B, T).
            sequence_lens: Sequence lengths of shape (B,).
            kv_cache: Optional list of KVCache for each layer.
        Returns:
            logits: Output logits of shape (B, T, vocab_size)
            out_cache: Optional list of KVCache for each layer if kv_cache was provided.
        """
        return self(params, x=x, sequence_lens=sequence_lens, kv_cache=kv_cache, apply_lm_head=apply_lm_head)

    @property
    def sequence_len(self) -> int:
        return self.model.seq_len

    @property
    def activation_dtype(self):
        return convert_dtype(self.model.activation_dtype)
