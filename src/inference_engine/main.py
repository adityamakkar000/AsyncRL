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

from src.data.config import Sample
from src.model import KVCache, Model

from .config import (
    InferenceConfig,
    InferenceResults,
    InferenceRollout,
    InferenceShardings,
    InferenceState,
)
from .utils import _maybe_force_eos, _maybe_force_eot, naive_sample

AXIS_NAME = "data"
PADDING_BUFFER = 1024


INTERUPT_THINKING_PHARSE = "Okay, time is up. Let me stop thinking and formulate a final answer now. \n\n</think>"


def apply_annealing(tokens: list[int], percent: float, max_seq_len: int) -> list[int]:
    return tokens[
        : max(1, int(len(tokens) * percent))
    ]  # TODO: fix this mathematically, if solution too long then it will be truncated


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

        if self.config.precompile:
            self.precompile_decode(params)
            self.precompile_prefill(params)

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

    def precompile_prefill(self, params: PyTree) -> None:
        """
        Precompile prefill function for different sequence lengths up to max_seq_len.
        The structure self.precompiled_dict = dict[seq_len --> compiled_fn].
        """

        curr_seq_len = self.config.intial_sequence_len
        params_host = self.setup_parameters(params)

        with jax.set_mesh(self.shardings.mesh):
            key = jax.random.PRNGKey(0)
            while curr_seq_len <= self.config.max_prefill_sequence_len:
                seq_lens_np = np.ones((1,), dtype=np.int32)
                x_init_np = np.ones((1, curr_seq_len), dtype=np.int32)

                x_init, seq_lens, params_sharded, key_sharded = self.put_batch_on_device(
                    x_init_np, seq_lens_np, params_host, key
                )

                self.precompile_dict["prefill"][curr_seq_len] = jax.jit(
                    self.prefill,
                    **self.shardings.prefill_shardings,
                )

                _output = self.precompile_dict["prefill"][curr_seq_len](x_init, seq_lens, params_sharded, key_sharded)

                curr_seq_len *= 2
            del params_host
            logger.info("Finished prefill precompile")

    def precompile_decode(self, params: PyTree) -> None:
        """
        Precompile decode function for max attention length only since we use jax.lax.while loop.
        """
        params_host = self.setup_parameters(params)

        with jax.set_mesh(self.shardings.mesh):
            params_sharded = jax.tree.map(self.replicate_across_axis, params_host)

            @partial(jax.jit, out_shardings=self.shardings.state_sharding)
            def create_initial_state() -> InferenceState:
                state = InferenceState(
                    next_token=jnp.ones((self.decode_size, 1), dtype=jnp.int32),
                    kv_cache=self.model.init_kv_cache(
                        self.decode_size,
                        length=self.max_attention_length,
                        dtype=self.config.kv_cache_dtype,
                        sharding=self.shardings.kv_cache_sharding,
                    ),
                    key=jax.random.PRNGKey(0),
                    seq_lens=jnp.array([1] * self.decode_size, dtype=jnp.int32),
                    stop_mask=jnp.zeros((self.decode_size, 1), dtype=bool),
                    end_of_think=jnp.zeros((self.decode_size, 1), dtype=bool),
                    out_tokens=jnp.ones((self.decode_size, self.max_attention_length), dtype=jnp.int32),
                    out_logprobs=jnp.zeros((self.decode_size, self.max_attention_length), dtype=jnp.float32),
                )
                return state

            state = create_initial_state()

            self.precompile_dict["decode"][self.max_attention_length] = jax.jit(
                self._decode_loop,
                donate_argnums=(0,),
                **self.shardings.decode_shardings,
            )
            _output = self.precompile_dict["decode"][self.max_attention_length](state, params_sharded)

            del (_output, params_sharded, params_host)
            logger.info("Finished decode precompile")

    def compute_max_power_of_two(self, n: int, upper_bound: int) -> int:
        """Compute the maximum power of two less than or equal to n and upper_bound."""
        return min(1 << (n.bit_length()), upper_bound)

    def compute_max_padding_length(self, seq_lens: np.ndarray) -> int:
        """Compute the maximum padding length for the input batch based on the sequence lengths and the maximum sequence length."""
        return self.compute_max_power_of_two(max(seq_lens).item(), self.config.max_seq_len)

    def prepare_prompt(self, text: str, annealing_trace: str, annealing_percentage: float) -> list[int]:
        assert annealing_percentage != -1.0, "Annealing percentage must be provided"

        annealed_base = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": apply_prompt_template(text)},
                {"role": "assistant", "content": ""},
            ],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        ).split("\n</think>\n")[0]

        chat_base = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": apply_prompt_template(text)}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )

        if not annealing_trace or annealing_percentage <= 1e-6:
            annealed_template = chat_base
        else:
            trace_tokens = self.tokenizer.encode(annealing_trace, add_special_tokens=False)
            annealed_trace_str = self.tokenizer.decode(
                apply_annealing(trace_tokens, annealing_percentage, self.config.max_seq_len)
            )
            annealed_template = annealed_base + annealed_trace_str

        # TODO: implement this after consulting with @adityamakkar000
        # self.tokenizer.apply_chat_template(
        #         get_chat_template(self.config.system_prompt, text),
        #         add_generation_prompt=True,
        #         enable_thinking=self.config.think_mode,
        #         tokenize=True,
        #     )
        #     for text in texts

        return self.tokenizer.encode(annealed_template, add_special_tokens=False)

    def tokenize(
        self,
        texts: list[str],
        annealing_traces: list[str],
        annealing_percentages: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Tokenize the input texts and pad them to the maximum sequence length in the batch.
        Args:
            texts (list[str]): The list of input texts to tokenize.
            annealing_traces (list[str]): The list of annealing traces to tokenize.
            annealing_percentages (list[float]): The list of annealing percentages to tokenize.
        Returns:
            tokens (Array): The tokenized and padded input texts. Shape: [batch_size, max_seq_len].
            seq_lens (Array): The original sequence lengths before padding. Shape: [batch_size].
        """
        assert len(texts) == len(annealing_traces) == len(annealing_percentages), (
            f"Lengths of texts, annealing_traces, and annealing_percentages must be equal, got {len(texts)}, {len(annealing_traces)}, {len(annealing_percentages)}"
        )

        inputs: list[list[int]] = [
            self.prepare_prompt(text, annealing_trace, annealing_percentage)
            for text, annealing_trace, annealing_percentage in zip(texts, annealing_traces, annealing_percentages)
        ]

        seq_lens = np.array([len(x) for x in inputs], dtype=np.int32)
        padding_length = max(self.compute_max_padding_length(seq_lens), self.config.intial_sequence_len)
        inputs = [(padding_length - len(x)) * [self.tokenizer.pad_token_id] + x for x in inputs]

        # for i, x in enumerate(inputs):
        #     if not isinstance(x, list):
        #         print("not a list", i, type(x), x)
        #     elif not all(isinstance(t, (int, np.integer)) for t in x):
        #         print("bad token list", i, type(x), x[:5], [type(t) for t in x[:5]])
        #     else:
        #         print(i, len(x))
        tokens = np.array(inputs, dtype=np.int32)

        if (T := tokens.shape[1]) > PADDING_BUFFER:
            raise ValueError(
                f"Input sequence padded prompts (T={T}) was greater than kv-cache length with padding, either implement roll cache or add additional buffer space"
            )

        return tokens, seq_lens

    def cleanup_rollouts(self, rollouts: list[InferenceRollout]) -> list[InferenceRollout]:
        """
        Remove padding tokens and trailing repeated EOS tokens from rollouts.
        Keeps the first EOS token and truncates everything after it.
        """
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
        """
        Detokenize the output rollouts into strings.
        Args:
            tokens list[InferenceRollout]: The list of rollouts to detokenize.
        Returns:
            list[list[str]]: The detokenized output strings.
        """
        if isinstance(tokens, InferenceRollout):
            tokens = [tokens]

        output_strs = []
        for rollout in tokens:
            rollout_strs = self.tokenizer.batch_decode(rollout.rollouts, skip_special_tokens=False)
            output_strs.append(rollout_strs)

        return output_strs

    def prefill(self, input_tokens: Array, seq_lens: Array, params: PyTree, key: Array) -> InferenceState:
        """
        Run model forward pass for prefill
        Args:
            input_tokens (Array): The input tokens for the prefill step. Shape: [decode_size, seq_len].
            seq_lens (Array): The sequence lengths of the input tokens. Shape: [decode_size].
            params (PyTree): The model parameters.
            key (Array): The random key for any stochastic operations during prefill.
        Returns:
            InferenceState: The state after the prefill step, containing the filled kv cache and other necessary information for decoding.
        """
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
                params,
                x=input_tokens[:, :-1],
                sequence_lens=seq_lens - 1,
                kv_cache=kv_cache,
            )
            out_tokens = jnp.ones((1, self.max_attention_length), dtype=jnp.int32) * self.tokenizer.eos_token_id
            out_logprobs = jnp.zeros((1, self.max_attention_length), dtype=jnp.float32)
            out_tokens = jax.lax.dynamic_update_slice_in_dim(out_tokens, input_tokens, 0, axis=1)
            out_logprobs = jax.lax.dynamic_update_slice_in_dim(
                out_logprobs,
                -jnp.inf * jnp.ones_like(input_tokens, dtype=jnp.float32),
                0,
                axis=1,
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
        """
        Perform the prefill step with precompilation for the given input tokens and sequence lengths.
        Args:
            input_tokens (Array): The input tokens for the prefill step. Shape: [batch_size, seq_len].
            seq_lens (Array): The sequence lengths of the input tokens. Shape: [batch_size].
            params (PyTree): The model parameters.
            key (Array): The random key for any stochastic operations during prefill.
        Returns:
            InferenceState: The state after the prefill step, containing the filled kv cache and other necessary information for decoding.
            dict[str, float]: Metrics collected during the prefill step, such as time taken.
        """
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
        """
        Perform the decode step of the inference, which runs the model for one step to get the next token and update the kv cache.
        Args:
            state (InferenceState): The current state of the inference, containing the current token, kv cache, and other necessary information.
            params (PyTree): The model parameters.
        Returns:
            InferenceState: The updated state after the decode step, containing the next token, updated kv cache, and other necessary information for the next step.
        """
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

    def single_rollout(self, state: InferenceState, params: PyTree) -> tuple[Array, Array, dict[str, float]]:
        """
        Perform a single rollout for the given inference state using an on-device while_loop.
        Args:
            state (InferenceState): The initial state of the inference, containing the current token, kv cache, and other necessary information.
            params (PyTree): The model parameters.
        Returns:
            Array: The output tokens generated during the rollout. Shape: [decode_size, total_seq_len].
            Array: The log probabilities of the generated tokens during the rollout. Shape: [decode_size, total_seq_len].
            dict[str, float]: Metrics collected during the rollout, such as time taken and tokens per second.
        """
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
                lambda x: list(jax.device_get(x)),
                (state.out_tokens, state.out_logprobs),
            )

        n_steps = (state.kv_cache[0].length - initial_cache_length).item()
        total_tokens = n_steps * self.decode_size * jax.process_count()
        total_time = t.data["time"]

        decode_metrics = {
            "total_time": total_time,
            "total_decode_tokens": total_tokens,
            "decode_steps": n_steps,
            "tps": total_tokens / total_time,
            "sps": n_steps / total_time,
        }

        logger.info(
            f"Inferenced {decode_metrics['decode_steps']} tokens in {decode_metrics['total_time']:.2f} seconds ({decode_metrics['tps']:.2f} tps, {decode_metrics['sps']:.2f} sps)"
        )
        return (
            out_tokens,
            out_logprobs,
            decode_metrics,
        )

    def rollout_group(self, x: Array, seq_lens: Array, params: PyTree, key: Array) -> tuple[InferenceRollout, PyTree]:
        """
        Perform rollouts for a group of inputs, where the group size is determined by the config.
        Args:
            x (Array): The input tokens for the group. Shape: [batch_size, seq_len].
            seq_lens (Array): The sequence lengths of the input tokens. Shape: [batch_size].
            params (PyTree): The model parameters.
            key (Array): The random key for any stochastic operations during the rollouts.
        Returns:
            InferenceRollout: The rollouts generated for the group, containing the output tokens and log probabilities.
            PyTree: Metrics collected during the rollouts, such as time taken and tokens per second.
        """
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
        """
        Perform rollouts for the entire batch of inputs by splitting them into groups and running rollouts for each group.
        Args:
            batch_tokens (Array): The input tokens for the entire batch. Shape: [batch_size, seq_len].
            seq_lens (Array): The sequence lengths of the input tokens. Shape: [batch_size].
            key (Array): The random key for any stochastic operations during the rollouts.
            params (PyTree): The model parameters.
        Returns:
            list[InferenceRollout]: The rollouts generated for the entire batch, containing the output tokens and log probabilities for each input.
            PyTree: Metrics collected during the rollouts, such as time taken and tokens per second.
        """
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
        """
        Prepare the random key and model parameters for multi-host inference by performing an all-gather across hosts.
        Args:
            key (Array): The random key for any stochastic operations during inference.
            params (PyTree): The model parameters to use for inference.
        Returns:
            Array: The updated random key after folding in the host index.
            PyTree: The model parameters after performing an all-gather across hosts.
        """
        key = jax.device_get(jax.random.fold_in(key, stax.get_rank()))
        params = self.setup_parameters(params)
        return key, params

    def __call__(self, samples: list[Sample], key: Array, params: PyTree) -> InferenceResults:
        """
        Perform inference for the given input prompts, random key, and model parameters.
        Args:
            samples (list[Sample]): The list of input samples to perform inference on.
            key (Array): The random key for any stochastic operations during inference.
            params (PyTree): The model parameters to use for inference.
            detokenize (bool): Whether to detokenize the output rollouts into strings. Default is False.
        Returns:
            InferenceResults: The results of the inference, containing the output rollouts, optionally the detokenized output strings, and any collected metrics.
        """
        with Tracker(timer=True) as t:
            key, params = self.multihost_prep(key, params)
            inp_tokens, seq_lens = self.tokenize(
                texts=[sample.prompt for sample in samples],
                annealing_traces=[sample.solution if sample.solution is not None else "" for sample in samples],
                annealing_percentages=[
                    (sample.annealing_percentage if sample.annealing_percentage is not None else -1.0)
                    for sample in samples
                ],  # can be used for error check
            )
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
