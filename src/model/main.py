import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import stax
from flax import linen as nn
from hydra.utils import instantiate
from jax.sharding import Sharding, SingleDeviceSharding
from jaxtyping import Array, PyTree
from omegaconf import DictConfig
from optax import GradientTransformation
from stax import HFModelBase
from stax import staxLogger as logger

from .config import BaseModel, KVCache, ModelConfig
from .utils import convert_dtype, fused_linear_selection, get_lm_head_weights

shardingType = PyTree[Sharding] | None


class Model(HFModelBase):
    def __init__(self, config: DictConfig | ModelConfig):
        self.config = config
        self.model: BaseModel = instantiate(config.model_args)
        assert isinstance(self.model, BaseModel), (
            f"Expected model to be an instance of BaseModel, got {type(self.model)}"
        )

    def init_state(
        self, rng: Array, tx: GradientTransformation | None, *, sharding: shardingType = None, abstract: bool = False
    ) -> PyTree:
        x_init = jnp.ones((1, self.sequence_len), dtype=jnp.int32)
        seq_lens = jnp.array([1])

        def init_state(rng, x_init, sequence_lens):
            params = jax.eval_shape(self.model.init, rngs=rng, x=x_init, sequence_lens=sequence_lens, kv_cache=None)[
                "params"
            ]
            out_state = {"params": params}
            if tx:
                out_state["opt_state"] = tx.init(jax.tree.map(lambda x: jnp.empty(x.shape, dtype=x.dtype), params))
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

    def init_kv_cache(self, batch_size: int, length: int, sharding: KVCache, dtype: str = "bfloat16") -> list[KVCache]:
        if length > self.sequence_len:
            raise ValueError(f"Requested KV cache length {length} exceeds maximum of {self.sequence_len + 1024}")

        n_layers, *kv_shape = self.model.kv_shape

        @jax.jit
        def _init():
            def zeros(out_sharding):
                return jnp.zeros(
                    (
                        batch_size,
                        length,
                        *kv_shape,
                    ),
                    dtype=convert_dtype(dtype),
                    out_sharding=out_sharding,
                )

            return KVCache(
                k=zeros(sharding.k),
                v=zeros(sharding.v),
                length=jnp.zeros((batch_size,), dtype=jnp.int32, out_sharding=sharding.length),  # type: ignore
            )

        return [_init() for _ in range(n_layers)]

    def load_from_ckpt(
        self, path: str, step_number: int | None = None, use_best=False
    ) -> tuple[int, PyTree, dict[str, float]]:
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
        step_number: int | None = None,
    ) -> tuple[int, PyTree, dict[str, float]]:
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
