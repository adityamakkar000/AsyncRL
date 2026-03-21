from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import stax
from jax.experimental.multihost_utils import process_allgather, sync_global_devices
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree
from stax import Tracker
from stax.logger import staxLogger as logger
from transformers import AutoTokenizer

from src.model import KVCache, Model

from .config import InferenceConfig, InferenceResults, InferenceRollout, InferenceShardings, InferenceState
from .utils import _maybe_force_eos, _maybe_force_eot, naive_sample

AXIS_NAME = "data"
PADDING_BUFFER = 1024


INTERUPT_THINKING_PHARSE = "Okay, time is up. Let me stop thinking and formulate a final answer now. \n\n</think>"


def apply_prompt_template(text: str) -> str:
    return f"""Solve the following math problem step by step. Put your answer inside \\boxed{{}}.
{text}
Remember to put your answer inside \\boxed{{}}."""


def apply_system_prompt_template() -> str:
    return r"""You are a helpful AI assistant.
For every problem, you must reason step-by-step inside <think></think> tags before giving the final answer.
"""


def get_chat_template(system_prompt: bool, text: str) -> list[dict[str, str]]:
    if system_prompt:
        return [
            {"role": "system", "content": apply_system_prompt_template()},
            {"role": "user", "content": apply_prompt_template(text)},
        ]
    else:
        return [{"role": "user", "content": apply_prompt_template(text)}]


class InferenceEngine:
    """
    Inference Engine for RLVR that performs static batching with group-based rollouts, supporting multi-host decoding
    """

    def __init__(self, model: Model, params: PyTree, config: InferenceConfig):
        self.model = model
        self.config = config
        self.validate_config()

        self.decode_size = config.batch_size * config.n_replicas
        self.shardings = self.get_shardings(params)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.hf_model_name)
        self.precompile_dict = {
            "prefill": {},
            "decode": {},
        }

        self.thinking_tokens = (
            self.tokenizer(INTERUPT_THINKING_PHARSE, add_special_tokens=False, return_tensors="np")
            .input_ids[0]
            .tolist()
        )

    def validate_config(self):
        """Validate the inference configuration to ensure it meets the requirements for the inference engine."""
        assert self.config.n_replicas <= jax.local_device_count(), (
            f"Number of replicas {self.config.n_replicas} must be less than or equal to number of devices {jax.device_count()}"
        )
        assert self.config.max_seq_len <= self.model.sequence_len, (
            f"expected inference max seq len {self.config.max_seq_len} to be less than model sequence length {self.model.sequence_len}"
        )
        assert self.config.max_seq_len & (self.config.max_seq_len - 1) == 0, (
            f"max_seq_len must be a power of 2, got {self.config.max_seq_len}"
        )
        assert self.config.intial_sequence_len & (self.config.intial_sequence_len - 1) == 0, (
            f"initial_sequence_len must be a power of 2, got {self.config.intial_sequence_len}"
        )
        assert self.config.group_size % (self.config.batch_size * self.config.n_replicas) == 0, (
            "Batch size must be divisible by group size for static batching"
        )

        assert self.config.max_prefill_sequence_len <= self.config.max_seq_len, (
            f"max_prefill_sequence_len {self.config.max_prefill_sequence_len} must be less than or equal to max_seq_len {self.config.max_seq_len}"
        )
        assert self.config.max_prefill_sequence_len & (self.config.max_prefill_sequence_len - 1) == 0, (
            f"max_prefill_sequence_len must be a power of 2, got {self.config.max_prefill_sequence_len}"
        )

        if self.config.reasoning_budget is not None:
            answer_tokens = min(1024, self.config.max_seq_len // 2)
            assert self.config.reasoning_budget <= (self.config.max_seq_len - answer_tokens), (
                f"Reasoning budget {self.config.reasoning_budget} must be less than or equal to {self.config.max_seq_len - answer_tokens} to account answer tokens"
            )

        if not self.config.think_mode:
            assert self.config.reasoning_budget is None, "Reasoning budget should be None when think_mode is disabled"

        if self.config.top_k is not None:
            assert self.config.top_k > 0, f"top_k must be positive, got {self.config.top_k}"

        if self.config.top_p is not None:
            assert 0.0 < self.config.top_p <= 1.0, f"top_p must be in the range (0, 1], got {self.config.top_p}"

    def get_shardings(self, params) -> InferenceShardings:
        """Get the shardings for the model parameters, kv cache, and inference state based on the configuration."""
        local_devices = np.array(jax.local_devices())
        mesh = jax.make_mesh((self.config.n_replicas,), (AXIS_NAME,), devices=local_devices[: self.config.n_replicas])  # type: ignore

        replicate_sharding = jax.NamedSharding(mesh, P())
        split_sharding = jax.NamedSharding(mesh, P(AXIS_NAME))

        kv_sharding = KVCache(
            k=split_sharding,  # type: ignore
            v=split_sharding,  # type: ignore
            length=replicate_sharding,  # type: ignore
        )

        state_sharding = InferenceState(
            next_token=split_sharding,  # type: ignore
            kv_cache=[kv_sharding for _ in range(self.model.config.qwen_config.n_layers)],
            key=replicate_sharding,  # type: ignore
            seq_lens=split_sharding,  # type: ignore
            stop_mask=split_sharding,  # type: ignore
            end_of_think=split_sharding,  # type: ignore
            out_tokens=split_sharding,  # type: ignore
            out_logprobs=split_sharding,  # type: ignore
        )

        params_sharding = jax.tree.map(lambda _p: replicate_sharding, params)

        prefill_shardings = {
            "in_shardings": (
                replicate_sharding,
                replicate_sharding,
                replicate_sharding,
                replicate_sharding,
            ),
            "out_shardings": state_sharding,
        }
        decode_shardings = {
            "in_shardings": (state_sharding, params_sharding),
            "out_shardings": state_sharding,
        }

        return InferenceShardings(
            mesh=mesh,
            split_sharding=split_sharding,
            replicate_sharding=replicate_sharding,
            kv_cache_sharding=kv_sharding,
            state_sharding=state_sharding,
            params_sharding=params_sharding,
            prefill_shardings=prefill_shardings,
            decode_shardings=decode_shardings,
        )

    def replicate_across_axis(self, value: Array) -> Array:
        return jax.device_put(value, self.shardings.replicate_sharding)

    def split_across_axis(self, value: Array) -> Array:
        return jax.device_put(value, self.shardings.split_sharding)

    def put_state_on_device(self, state: InferenceState) -> InferenceState:
        return jax.tree.map(lambda x, s: jax.device_put(x, s), state, self.shardings.state_sharding)

    def put_batch_on_device(
        self, batch: np.ndarray, seq_lens: np.ndarray, params: PyTree, key: Array
    ) -> tuple[Array, Array, PyTree, Array]:
        batch: Array = jax.make_array_from_callback(
            shape=batch.shape,
            sharding=self.shardings.replicate_sharding,
            data_callback=lambda _index: batch,  # index not needed
            dtype=batch.dtype,
        )
        seq_lens: Array = jax.make_array_from_callback(
            shape=seq_lens.shape,
            sharding=self.shardings.replicate_sharding,
            data_callback=lambda _index: seq_lens,  # index not needed
            dtype=seq_lens.dtype,
        )

        key = self.replicate_across_axis(key)
        params = jax.tree.map(self.replicate_across_axis, params)
        return batch, seq_lens, params, key

    def setup_parameters(self, params: PyTree) -> PyTree:
        return jax.tree.map(lambda p: process_allgather(p, tiled=True), params)

    def compute_max_power_of_two(self, n: int, upper_bound: int) -> int:
        """Compute the maximum power of two less than or equal to n and upper_bound."""
        return min(1 << (n.bit_length()), upper_bound)

    def compute_max_padding_length(self, seq_lens: np.ndarray) -> int:
        """Compute the maximum padding length for the input batch based on the sequence lengths and the maximum sequence length."""
        return self.compute_max_power_of_two(max(seq_lens).item(), self.config.max_seq_len)

    def tokenize(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        inputs: list[list[int]] = [
            self.tokenizer.apply_chat_template(
                get_chat_template(self.config.system_prompt, text),
                add_generation_prompt=True,
                enable_thinking=self.config.think_mode,
                tokenize=True,
            )
            for text in texts
        ]

        seq_lens = np.array([len(x) for x in inputs], dtype=np.int32)
        padding_length = max(self.compute_max_padding_length(seq_lens), self.config.intial_sequence_len)
        inputs = [(padding_length - len(x)) * [self.tokenizer.pad_token_id] + x for x in inputs]
        tokens = np.array(inputs, dtype=np.int32)

        if (T := tokens.shape[1]) > PADDING_BUFFER:
            raise ValueError(
                f"Input sequence padded prompts (T={T}) was greater than kv-cache length with padding, either implement roll cache or add additional buffer space"
            )

        return tokens, seq_lens

    def cleanup_rollouts(self, rollouts: list[InferenceRollout]) -> list[InferenceRollout]:
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        def clean_sequence(tokens: np.ndarray, logprobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            non_pad = tokens != pad_id
            tokens = tokens[non_pad]
            logprobs = logprobs[non_pad]

            eos_positions = np.where(tokens != eos_id)[0]
            cutoff = eos_positions[-1] + 2  # keep one token after eos
            tokens = tokens[:cutoff]
            logprobs = logprobs[:cutoff]

            return tokens, logprobs

        cleaned = []
        for rollout in rollouts:
            new_rollouts = []
            new_logprobs = []
            for tokens, lps in zip(rollout.rollouts, rollout.logprobs):
                t, lp = clean_sequence(np.asarray(tokens), np.asarray(lps))
                new_rollouts.append(t)
                new_logprobs.append(lp)
            cleaned.append(InferenceRollout(rollouts=new_rollouts, logprobs=new_logprobs))

        return cleaned

    def detokenizer(self, tokens: list[InferenceRollout] | InferenceRollout) -> list[list[str]]:
        if isinstance(tokens, InferenceRollout):
            tokens = [tokens]

        output_strs = []
        for rollout in tokens:
            rollout_strs = self.tokenizer.batch_decode(rollout.rollouts, skip_special_tokens=False)
            output_strs.append(rollout_strs)

        return output_strs

    def prefill(self, input_tokens: Array, seq_lens: Array, params: PyTree, key: Array) -> InferenceState:
        logger.info(f"Compiling prefill for sequence length {input_tokens.shape[1]}")

        with jax.named_scope("prefill"):
            kv_cache = self.model.init_kv_cache(
                1,
                length=self.max_attention_length,
                dtype=self.config.kv_cache_dtype,
                sharding=KVCache(
                    k=self.shardings.replicate_sharding,  # type: ignore
                    v=self.shardings.replicate_sharding,  # type: ignore
                    length=self.shardings.replicate_sharding,  # type: ignore
                ),
            )
            logits, out_cache = self.model.apply(
                params, x=input_tokens[:, :-1], sequence_lens=seq_lens - 1, kv_cache=kv_cache
            )
            out_tokens = jnp.ones((1, self.max_attention_length), dtype=jnp.int32) * self.tokenizer.eos_token_id
            out_logprobs = jnp.zeros((1, self.max_attention_length), dtype=jnp.float32)
            out_tokens = jax.lax.dynamic_update_slice_in_dim(out_tokens, input_tokens, 0, axis=1)
            out_logprobs = jax.lax.dynamic_update_slice_in_dim(
                out_logprobs, -jnp.inf * jnp.ones_like(input_tokens, dtype=jnp.float32), 0, axis=1
            )

        def bc_to_decode(x: Array) -> Array:
            return jnp.broadcast_to(x, (self.decode_size,) + x.shape[1:])

        return InferenceState(
            next_token=bc_to_decode(input_tokens[:, -1:]),
            seq_lens=bc_to_decode(seq_lens),
            kv_cache=[KVCache(k=bc_to_decode(kv.k), v=bc_to_decode(kv.v), length=kv.length) for kv in out_cache],
            key=key,
            stop_mask=jnp.zeros((self.decode_size, 1), dtype=bool),
            end_of_think=jnp.zeros((self.decode_size, 1), dtype=bool),
            out_tokens=bc_to_decode(out_tokens),
            out_logprobs=bc_to_decode(out_logprobs),
        )

    def prefill_step(
        self, input_tokens: Array, seq_lens: Array, params: PyTree, key: Array
    ) -> tuple[InferenceState, dict[str, float]]:
        precompiled_length = input_tokens.shape[1]
        if self.precompile_dict["prefill"].get(precompiled_length) is None:
            self.precompile_dict["prefill"][precompiled_length] = jax.jit(
                self.prefill,
                **self.shardings.prefill_shardings,
            )

        with Tracker(timer=True) as t:
            out: InferenceState = self.precompile_dict["prefill"][precompiled_length](
                input_tokens, seq_lens, params, key
            )
            jax.tree.map(lambda x: x.block_until_ready(), out)
        return out, {"ttft": t.data["time"]}

    def decode(self, state: InferenceState, params: PyTree) -> InferenceState:
        logger.info(f"Compiling decode step for attention length {state.kv_cache[0].k.shape[1]}")
        key, sample_key = jax.random.split(state.key)

        with jax.named_scope("fwd_pass"):
            logits, out_cache = self.model.apply(
                params,
                x=state.next_token,
                sequence_lens=state.seq_lens,
                kv_cache=state.kv_cache,
            )
        with jax.named_scope("sampling"):
            next_token, next_log_prob = naive_sample(
                logits,
                sample_key,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )

        with jax.named_scope("stop_masking"):
            end_of_think = state.end_of_think
            if self.config.reasoning_budget is not None:
                next_token, next_log_prob, end_of_think = _maybe_force_eot(
                    next_token,
                    next_log_prob,
                    state.end_of_think,
                    state.seq_lens,
                    reasoning_budget=self.config.reasoning_budget,
                    token_sequence=self.thinking_tokens,
                )

            next_token, next_log_prob, stop_mask = _maybe_force_eos(
                next_token,
                next_log_prob,
                state.stop_mask,
                state.seq_lens,
                max_seq_len=self.config.max_seq_len,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        out_tokens = jax.lax.dynamic_update_index_in_dim(state.out_tokens, next_token, state.kv_cache[0].length, axis=1)
        out_logprobs = jax.lax.dynamic_update_index_in_dim(
            state.out_logprobs, next_log_prob, state.kv_cache[0].length, axis=1
        )

        return InferenceState(
            next_token=next_token,
            kv_cache=out_cache,
            key=key,
            seq_lens=state.seq_lens + jnp.where(stop_mask, 0, 1)[:, 0],
            stop_mask=stop_mask,
            end_of_think=end_of_think,
            out_tokens=out_tokens,
            out_logprobs=out_logprobs,
        )

    def _decode_loop(self, state: InferenceState, params: PyTree) -> InferenceState:
        return jax.lax.while_loop(
            lambda state: ~jnp.all(state.stop_mask),
            lambda state: self.decode(state, params),
            state,
        )

    def _decode_single_loop(self, state: InferenceState, params: PyTree) -> InferenceState:
        return jax.lax.while_loop(
            lambda state: ~jnp.any(state.stop_mask),
            lambda state: self.decode(state, params),
            state,
        )

    def continuous_batch(self, prompts: InferenceState, params: PyTree):
        P = prompts.next_token.shape[0]
        prompts_list = [prompts.get_index(i) for i in range(P)]

        prompt_queue = []
        for prompt in prompts_list:
            for _ in range(self.config.group_size):
                prompt_queue.append(prompt)

        def create_batch(n) -> InferenceState:
            batch = [prompt_queue.pop() for _ in range(n)]
            return jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *batch)

        def roll_cache_to_length(state: InferenceState, length: int) -> InferenceState:
            diff = length - state.kv_cache[0].length

            def roll_kv_cache(kv: KVCache) -> KVCache:
                rolled_k = jnp.roll(kv.k, diff, axis=1)
                rolled_v = jnp.roll(kv.v, diff, axis=1)
                return KVCache(k=rolled_k, v=rolled_v, length=length)

            return state.replace(kv_cache=[roll_kv_cache(kv) for kv in state.kv_cache])

        def substitute_batch(state: InferenceState, new_batch: InferenceState) -> InferenceState:
            current_length = state.kv_cache[0].length
            new_batch = roll_cache_to_length(new_batch, current_length)
            state = InferenceState(
                next_token=jnp.where(state.stop_mask, new_batch.next_token, state.next_token),
                kv_cache=[
                    KVCache(
                        k=jnp.where(state.stop_mask[:, None, None], new_batch.kv_cache[i].k, state.kv_cache[i].k),
                        v=jnp.where(state.stop_mask[:, None, None], new_batch.kv_cache[i].v, state.kv_cache[i].v),
                        length=current_length,
                    )
                    for i in range(len(state.kv_cache))
                ],
                key=state.key,
                seq_lens=jnp.where(state.stop_mask, new_batch.seq_lens, state.seq_lens),
                stop_mask=jnp.where(state.stop_mask, new_batch.stop_mask, state.stop_mask),
                end_of_think=jnp.where(state.stop_mask, new_batch.end_of_think, state.end_of_think),
                out_tokens=jnp.where(state.stop_mask, new_batch.out_tokens, state.out_tokens),
                out_logprobs=jnp.where(state.stop_mask, new_batch.out_logprobs, state.out_logprobs),
            )

            max_seq = jnp.max(state.seq_lens)
            shift_back = -(current_length - max_seq)
            state = state.replace(
                kv_cache=[
                    KVCache(k=jnp.roll(kv.k, shift_back, axis=1), v=jnp.roll(kv.v, shift_back, axis=1), length=max_seq)
                    for kv in state.kv_cache
                ]
            )

            return state

        finished_tokens = []
        finished_logprobs = []

        state = create_batch(self.decode_size)
        while len(prompt_queue) > 0:
            state = self._decode_single_loop(state, params)
            n_finished = jnp.sum(state.stop_mask)
            new_batch = create_batch(n_finished)

            tokens, logprobs = (
                state.out_tokens[jnp.where(state.stop_mask)],
                state.out_logprobs[jnp.where(state.stop_mask)],
            )
            finished_tokens.append(tokens)
            finished_logprobs.append(logprobs)

            state = substitute_batch(state, new_batch)

        state = self._decode_loop(state, params)

    def single_rollout(self, state: InferenceState, params: PyTree) -> tuple[Array, Array, dict[str, float]]:
        initial_cache_length = jnp.copy(state.kv_cache[0].length)

        if self.precompile_dict["decode"].get(self.max_attention_length) is None:
            self.precompile_dict["decode"][self.max_attention_length] = jax.jit(
                partial(self._decode_loop),
                donate_argnums=(0,),
                **self.shardings.decode_shardings,
            )
        with Tracker(timer=True) as t:
            state = self.precompile_dict["decode"][self.max_attention_length](state, params)
            out_tokens, out_logprobs = jax.tree.map(
                lambda x: list(jax.device_get(x)), (state.out_tokens, state.out_logprobs)
            )

        n_steps = (state.kv_cache[0].length - initial_cache_length).item()
        total_tokens = n_steps * self.decode_size * jax.process_count()
        decode_time = t.data["time"]

        decode_metrics = {
            "decode_time": decode_time,
            "total_decode_tokens": total_tokens,
            "decode_steps": n_steps,
            "tps": total_tokens / decode_time,
            "sps": n_steps / decode_time,
        }

        logger.info(
            f"Inferenced {decode_metrics['decode_steps']} tokens in {decode_metrics['decode_time']:.2f} seconds ({decode_metrics['tps']:.2f} tps, {decode_metrics['sps']:.2f} sps)"
        )
        return (
            out_tokens,
            out_logprobs,
            decode_metrics,
        )

    def rollout_group(self, x: Array, seq_lens: Array, params: PyTree, key: Array) -> tuple[InferenceRollout, PyTree]:
        assert (B := x.shape[0]) == 1, f"Expected batch size {self.decode_size}, got {B}"
        rollout_output = InferenceRollout(rollouts=[], logprobs=[])
        decode_metrics_collected = []
        prefill_metrics_collected = []

        for step in range(self.config.group_size // self.decode_size):
            prefill_key = jax.random.fold_in(key, step)
            # prefill_key = key

            # NOTE:
            # we manually do prefill on each step instead of reusing
            # since then we do not need to keep a copy of a kv cache
            # prefill is very fast and so this save 2x memory
            state, prefill_metrics = self.prefill_step(x, seq_lens, params, prefill_key)
            output_tokens, output_logprobs, decode_metrics = self.single_rollout(state, params)

            rollout_output.rollouts.extend(output_tokens)
            rollout_output.logprobs.extend(output_logprobs)

            decode_metrics_collected.append(decode_metrics)
            prefill_metrics_collected.append(prefill_metrics)

            del state  # free kv cache

        metrics: dict[str, float] = dict()
        for stage in [decode_metrics_collected, prefill_metrics_collected]:
            for key in stage[0].keys():
                values = jnp.array([m[key] for m in stage])
                metrics |= {
                    f"{key}_mean": jnp.mean(values).item(),
                    f"{key}_std": jnp.std(values).item(),
                    f"{key}_max": jnp.max(values).item(),
                    f"{key}_min": jnp.min(values).item(),
                }

        return rollout_output, metrics

    def batch_rollout(
        self, batch_tokens: np.ndarray, seq_lens: np.ndarray, key: Array, params: PyTree
    ) -> tuple[list[InferenceRollout], PyTree]:
        B, _ = batch_tokens.shape
        output: list[InferenceRollout] = []
        metrics: list[dict[str, float]] = []

        x_batch_sharded, seq_lens_sharded, params_sharded, key_sharded = self.put_batch_on_device(
            batch_tokens, seq_lens, params, key
        )

        for i in range(B):
            batch_key = jax.random.fold_in(key_sharded, i)
            # batch_key = key_sharded
            batch_output, batch_metrics = self.rollout_group(
                jax.lax.dynamic_index_in_dim(x_batch_sharded, i, axis=0),
                jax.lax.dynamic_index_in_dim(seq_lens_sharded, i, axis=0),
                params_sharded,
                batch_key,
            )
            output.append(batch_output)
            metrics.append(batch_metrics)

        metrics: dict[str, float] = jax.tree.map(lambda *x: sum(x) / len(x), *metrics)

        return self.cleanup_rollouts(output), metrics

    def multihost_prep(self, key: Array, params: PyTree) -> tuple[Array, PyTree]:
        key = jax.device_get(jax.random.fold_in(key, stax.get_rank()))
        params = self.setup_parameters(params)
        return key, params

    def __call__(self, prompts: list[str], key: Array, params: PyTree) -> InferenceResults:
        """
        Perform inference for the given input prompts, random key, and model parameters.
        Args:
            prompts (list[str]): The list of input prompts.
            key (Array): PRNG key for inference.
            params (PyTree): The model parameters to use for inference.
            detokenize (bool): Whether to detokenize the output rollouts into strings. Default is False.
        Returns:
            InferenceResults: The results of the inference, containing the output rollouts, optionally the detokenized output strings, and any collected metrics.
        """
        assert len(prompts) % self.config._max_prompts_decode == 0, (
            f"Number of prompts {len(prompts)} must be divisible by max_prompts_decode {self.config._max_prompts_decode}"
        )

        with Tracker(timer=True) as t:
            key, params = self.multihost_prep(key, params)
            inp_tokens, seq_lens = self.tokenize(prompts)
            # use inference engine mesh context not STAX context
            with jax.set_mesh(self.shardings.mesh):
                output_rollouts, metrics = self.batch_rollout(inp_tokens, seq_lens, key, params)
            output_strs = self.detokenizer(output_rollouts)
            sync_global_devices("inference_engine_sync")
        metrics |= {"total_inference_time": t.data["time"]}
        metrics = {f"inference_metrics/{k}": v for k, v in metrics.items()}
        return InferenceResults(rollouts=output_rollouts, output_strs=output_strs, metrics=metrics)

    @property
    def max_attention_length(self) -> int:
        return min(self.config.max_seq_len, self.model.sequence_len) + PADDING_BUFFER
