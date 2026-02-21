import time
from functools import partial

import jax
import jax.numpy as jnp
import stax
from jax.experimental.multihost_utils import process_allgather, sync_global_devices
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree
from stax import Tracker
from stax.logger import staxLogger as logger
from transformers import AutoTokenizer

from src.model import KVCache, Model, convert_dtype

from .config import InferenceConfig, InferenceResults, InferenceRollout, InferenceShardings, InferenceState

AXIS_NAME = "data"
LOG_EVERY_N_STEPS = 250


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

        if self.config.precompile:
            self.precompile_decode(params)
            self.precompile_prefill(params)

    def validate_config(self):
        """Validate the inference configuration to ensure it meets the requirements for the inference engine."""
        assert self.config.n_replicas <= jax.device_count(), (
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
        if self.config.top_k is not None:
            assert self.config.top_k > 0, f"top_k must be positive, got {self.config.top_k}"

        if self.config.top_p is not None:
            assert 0.0 < self.config.top_p <= 1.0, f"top_p must be in the range (0, 1], got {self.config.top_p}"

    def get_shardings(self, params) -> InferenceShardings:
        """Get the shardings for the model parameters, kv cache, and inference state based on the configuration."""
        mesh = jax.make_mesh((self.config.n_replicas,), (AXIS_NAME,))

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
            params=jax.tree.map(lambda _p: replicate_sharding, params),
            stop_mask=split_sharding,  # type: ignore
            end_of_think=split_sharding,  # type: ignore
            out_tokens=split_sharding,  # type: ignore
            out_logprobs=split_sharding,  # type: ignore
        )

        prefill_shardings = {
            "in_shardings": (
                split_sharding,
                split_sharding,
                replicate_sharding,
                replicate_sharding,
            ),
            "out_shardings": state_sharding,
        }
        decode_shardings = {
            "in_shardings": (state_sharding,),
            "out_shardings": state_sharding,
        }

        return InferenceShardings(
            split_sharding=split_sharding,
            replicate_sharding=replicate_sharding,
            kv_cache_sharding=kv_sharding,
            state_sharding=state_sharding,
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
        self, batch: Array, seq_lens: Array, params: PyTree, key: Array
    ) -> tuple[Array, Array, PyTree, Array]:
        batch, seq_lens = self.split_across_axis(batch), self.split_across_axis(seq_lens)
        key = self.replicate_across_axis(key)
        params = jax.tree.map(self.replicate_across_axis, params)
        return batch, seq_lens, params, key

    def setup_parameters(self, params: PyTree) -> PyTree:
        params_dtype = convert_dtype(self.config.params_dtype)
        params_host = jax.tree.map(lambda p: process_allgather(p.astype(params_dtype), tiled=True), params)
        return params_host

    def precompile_prefill(self, params: PyTree) -> None:
        """
        Precompile prefill function for different sequence lengths up to max_seq_len.
        The structure self.precompiled_dict = dict[seq_len --> compiled_fn].
        """

        curr_seq_len = self.config.intial_sequence_len

        key = jax.random.PRNGKey(0)
        seq_lens = jnp.array([1] * self.decode_size)
        params = self.setup_parameters(params)

        while curr_seq_len <= self.config.max_prefill_sequence_len:
            x_init = jnp.ones((self.decode_size, curr_seq_len), dtype=jnp.int32)

            x_init, seq_lens, params, key = self.put_batch_on_device(x_init, seq_lens, params, key)
            self.precompile_dict["prefill"][curr_seq_len] = jax.jit(
                self.prefill,
                **self.shardings.prefill_shardings,
            )
            _output = self.precompile_dict["prefill"][curr_seq_len](x_init, seq_lens, params, key)
            del _output
            curr_seq_len *= 2
        del params
        logger.info("Finished prefill precompile")

    def precompile_decode(self, params: PyTree) -> None:
        """
        Precompile decode function for different attention lengths up to max_seq_len.
        The structure self.precompiled_dict = dict[attention_len --> compiled_fn].
        """
        curr_size = self.config.intial_sequence_len

        def get_mock_state():
            state = InferenceState(
                next_token=jnp.ones((self.decode_size, 1), dtype=jnp.int32),
                kv_cache=self.model.init_kv_cache(
                    self.decode_size, dtype=self.config.kv_cache_dtype, sharding=self.shardings.kv_cache_sharding
                ),
                key=jax.random.PRNGKey(0),
                seq_lens=jnp.array([1] * self.decode_size),
                params=self.setup_parameters(params),
                stop_mask=jnp.zeros((self.decode_size, 1), dtype=bool),
                end_of_think=jnp.zeros((self.decode_size, 1), dtype=bool),
                out_tokens=jnp.ones((self.decode_size, self.model.sequence_len + 1024), dtype=jnp.int32),
                out_logprobs=jnp.zeros(
                    (self.decode_size, self.model.sequence_len + 1024), dtype=self.model.activation_dtype
                ),
            )

            return self.put_state_on_device(state)

        # for decode compile 2 * max len
        # since padding can be at most  2 * max_len
        # cache upper bound is self.model.sequence_len + 1024
        while curr_size <= min(2 * self.config.max_seq_len, self.model.sequence_len + 1024):
            self.precompile_dict["decode"][curr_size] = jax.jit(
                partial(self.decode, attention_length=curr_size),
                donate_argnums=(0,),
                **self.shardings.decode_shardings,
            )
            _output = self.precompile_dict["decode"][curr_size](get_mock_state())
            del _output
            curr_size *= 2
        logger.info("Finished decode precompile")

    def compute_max_power_of_two(self, n: int, upper_bound: int) -> int:
        """Compute the maximum power of two less than or equal to n and upper_bound."""
        return min(1 << (n.bit_length()), upper_bound)

    def compute_max_padding_length(self, seq_lens: Array) -> int:
        """Compute the maximum padding length for the input batch based on the sequence lengths and the maximum sequence length."""
        return self.compute_max_power_of_two(jnp.max(seq_lens).item(), self.config.max_seq_len)

    def compute_attention_length(self, cache_length: Array) -> int:
        """Compute the attention length for the decode step based on the current cache length."""
        # use the model attention len as the upper bound
        return self.compute_max_power_of_two(cache_length.item(), self.model.sequence_len)

    def tokenize(self, texts: list[str]) -> tuple[Array, Array]:
        """
        Tokenize the input texts and pad them to the maximum sequence length in the batch.
        Args:
            texts (list[str]): The list of input texts to tokenize.
        Returns:
            tokens (Array): The tokenized and padded input texts. Shape: [batch_size, max_seq_len].
            seq_lens (Array): The original sequence lengths before padding. Shape: [batch_size].
        """

        def apply_prompt_template(text: str) -> str:
            return f"""
                Solve the following math problem step by step. Put your answer inside \\boxed{{}}.
                {text}
                
                Remember to put your answer inside \\boxed{{}}.
            """

        inputs: list[list[int]] = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": apply_prompt_template(text)}],
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for text in texts
        ]
        seq_lens = jnp.array([len(x) for x in inputs], dtype=jnp.int32)
        padding_length = max(self.compute_max_padding_length(seq_lens), self.config.intial_sequence_len)
        assert padding_length <= self.config.max_prefill_sequence_len, (
            f"Computed padding length {padding_length} is greater than max_prefill_sequence_len {self.config.max_prefill_sequence_len}, either reduce input sequence lengths or increase max_prefill_sequence_len"
        )
        inputs = [(padding_length - len(x)) * [self.tokenizer.pad_token_id] + x for x in inputs]
        tokens = jnp.array(inputs)

        if (T := tokens.shape[1]) > 1024:
            raise ValueError(
                f"Input sequence padded prompts (T={T}) was greater than kv-cache length with padding, either implment roll cache or add additional buffer space"
            )

        return tokens, jnp.array(seq_lens, dtype=jnp.int32)

    def cleanup_rollouts(self, rollouts: list[InferenceRollout]) -> list[InferenceRollout]:
        """
        Remove padding and eos tokens from the rollouts.
        Args:
            rollouts (list[InferenceRollout]): The list of rollouts to clean up.
        Returns:
            list[InferenceRollout]: The cleaned up rollouts.
        """

        def remove_pad_token(tokens: Array, logprobs: Array) -> tuple[Array, Array]:
            pad_token_id = self.tokenizer.pad_token_id
            non_pad_mask = tokens != pad_token_id
            return tokens[non_pad_mask], logprobs[non_pad_mask]

        def remove_eos_token(tokens: Array, logprobs: Array) -> tuple[Array, Array]:
            eos_token_id = self.tokenizer.eos_token_id
            eos_index = jnp.where(tokens == eos_token_id)[0][:2]
            non_eos_mask = tokens != eos_token_id
            non_eos_mask[eos_index] = True
            return tokens[non_eos_mask], logprobs[non_eos_mask]

        def clean_rollout(rollout: InferenceRollout) -> InferenceRollout:
            rollouts = []
            logprobs = []
            for roll, log in zip(rollout.rollouts, rollout.logprobs):
                roll, log = remove_pad_token(roll, log)
                roll, log = remove_eos_token(roll, log)
                rollouts.append(roll)
                logprobs.append(log)
            return InferenceRollout(rollouts=rollouts, logprobs=logprobs)

        return [clean_rollout(rollout) for rollout in rollouts]

    def detokenizer(self, tokens: list[InferenceRollout] | InferenceRollout) -> list[list[str]]:
        """
        Detokenize the output rollouts into strings.
        Args:
            tokens (list[InferenceRollout] | InferenceRollout): The list of rollouts to detokenize.
        Returns:
            list[list[str]] | list[str]: The detokenized output strings.
        """
        if isinstance(tokens, InferenceRollout):
            tokens = [tokens]

        output_strs = []
        for rollout in tokens:
            rollout_strs = self.tokenizer.batch_decode(rollout.rollouts, skip_special_tokens=False)
            output_strs.append(rollout_strs)

        return output_strs

    def sample_logits(self, logits: Array, key: Array) -> tuple[Array, Array]:
        """
        Sample the next token from the logits using temperature, top-k, and top-p sampling.
        Args:
            logits (Array): The logits from the model. Shape: [batch_size, vocab_size].
            key (Array): The random key for sampling.
        Returns:
            next_tokens (Array): The sampled next tokens. Shape: [batch_size, 1].
            next_logprobs (Array): The log probabilities of the sampled tokens. Shape: [batch_size, 1].
        """
        B, T, V = logits.shape
        logits = logits[:, -1, :] / self.config.temperature

        if self.config.top_k:
            assert self.config.top_k < V, f"top_k must be less than vocab size {V}, got {self.config.top_k}"
            top_k = min(self.config.top_k, V)
            logits, base_indices = jax.lax.top_k(logits, top_k)
        else:
            base_indices = jnp.tile(jnp.arange(V), (B, 1))
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        if self.config.top_p:
            sort_idx = jnp.argsort(-log_probs, axis=-1)
            sorted_probs = jnp.take_along_axis(log_probs, sort_idx, axis=-1)
            sorted_logits = jnp.take_along_axis(logits, sort_idx, axis=-1)

            mask = jnp.cumsum(sorted_probs, axis=-1) <= self.config.top_p
            mask = mask.at[:, 0].set(True)

            filtered_logits = jnp.where(mask, sorted_logits, -jnp.inf)

            logits = jnp.take_along_axis(filtered_logits, jnp.argsort(sort_idx, axis=-1), axis=-1)
            log_probs = jax.nn.log_softmax(logits, axis=-1)

        next_idx = jax.random.categorical(key, logits, axis=-1)[:, None]
        next_tokens = jnp.take_along_axis(base_indices, next_idx, axis=-1)
        next_logprobs = jnp.take_along_axis(log_probs, next_idx, axis=-1)

        return next_tokens, next_logprobs

    def prefill(self, input_tokens: Array, seq_lens: Array, params: PyTree, key: Array) -> InferenceState:
        """
        Perform the prefill step of the inference, which runs the model on the input tokens to fill the kv cache.
        Args:
            input_tokens (Array): The input tokens for the prefill step. Shape: [decode_size, seq_len].
            seq_lens (Array): The sequence lengths of the input tokens. Shape: [decode_size].
            params (PyTree): The model parameters.
            key (Array): The random key for any stochastic operations during prefill.
        Returns:
            InferenceState: The state after the prefill step, containing the filled kv cache and other necessary information for decoding.
        """
        logger.info(f"Compiling prefill for sequence length {input_tokens.shape[1]}")
        kv_cache = self.model.init_kv_cache(
            self.decode_size, dtype=self.config.kv_cache_dtype, sharding=self.shardings.kv_cache_sharding
        )
        logits, out_cache = self.model.apply(
            params, x=input_tokens[:, :-1], sequence_lens=seq_lens - 1, kv_cache=kv_cache
        )
        out_tokens = (
            jnp.ones((self.decode_size, self.model.sequence_len + 1024), dtype=jnp.int32) * self.tokenizer.eos_token_id
        )
        out_logprobs = jnp.zeros((self.decode_size, self.model.sequence_len + 1024), dtype=self.model.activation_dtype)
        out_tokens = jax.lax.dynamic_update_slice_in_dim(out_tokens, input_tokens, 0, axis=1)
        out_logprobs = jax.lax.dynamic_update_slice_in_dim(
            out_logprobs, -jnp.inf * jnp.ones_like(input_tokens, dtype=self.model.activation_dtype), 0, axis=1
        )

        return InferenceState(
            next_token=input_tokens[:, -1:],
            seq_lens=seq_lens,
            kv_cache=out_cache,
            params=params,
            key=key,
            stop_mask=jnp.zeros((self.decode_size, 1), dtype=bool),
            end_of_think=jnp.zeros((self.decode_size, 1), dtype=bool),
            out_tokens=out_tokens,
            out_logprobs=out_logprobs,
        )

    def decode(self, state: InferenceState, attention_length: int) -> InferenceState:
        """
        Perform the decode step of the inference, which runs the model for one step to get the next token and update the kv cache.
        Args:
            state (InferenceState): The current state of the inference, containing the current token, kv cache, and other necessary information.
            attention_length (int): The attention length to use for this decode step, which determines how much of the past context the model attends to.
        Returns:
            InferenceState: The updated state after the decode step, containing the next token, updated kv cache, and other necessary information for the next step.
        NOTE: attention_length is meant to compiled statically
        """
        logger.info(f"Compiling decode for attention length {attention_length}")
        key, sample_key = jax.random.split(state.key)
        logits, out_cache = self.model.apply(
            state.params,
            x=state.next_token,
            sequence_lens=state.seq_lens,
            kv_cache=state.kv_cache,
            attention_len=attention_length,
        )
        next_token, next_log_prob = self.sample_logits(logits, sample_key)

        end_of_think = state.end_of_think | (next_token == 151668)
        if self.config.reasoning_budget is not None:
            budget_exceed = (self.config.reasoning_budget is not None) & (
                state.seq_lens[:, None] + 1 > self.config.reasoning_budget
            )
        else:
            budget_exceed = jnp.zeros_like(end_of_think, dtype=bool)
        interrupt_mask = budget_exceed & ~end_of_think
        end_of_think = end_of_think | interrupt_mask
        next_token = jnp.where(interrupt_mask, 151668, next_token)
        next_log_prob = jnp.where(interrupt_mask, 0.0, next_log_prob)

        # split two different masks since if we have <eos> naturally we want that logprob in the next_log_probs (eos_stop_mask)
        length_stop_mask = state.stop_mask | (state.seq_lens[:, None] + 1 > self.config.max_seq_len)
        eos_stop_mask = next_token == self.tokenizer.eos_token_id
        stop_mask = eos_stop_mask | length_stop_mask

        next_token = jnp.where(stop_mask, self.tokenizer.eos_token_id, next_token)
        next_log_prob = jnp.where(stop_mask, 0, next_log_prob)
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
            params=state.params,
            end_of_think=end_of_think,
            out_tokens=out_tokens,
            out_logprobs=out_logprobs,
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

    def decode_step(self, inference_state: InferenceState) -> InferenceState:
        """
        Perform the decode step with precompilation for the given inference state.
        Args:
            inference_state (InferenceState): The current state of the inference, containing the current token, kv cache, and other necessary information.
        Returns:
            InferenceState: The updated state after the decode step, containing the next token, updated kv cache, and other necessary information for the next step.
        """

        attention_length = self.compute_attention_length(inference_state.kv_cache[0].length)
        if self.precompile_dict["decode"].get(attention_length) is None:
            self.precompile_dict["decode"][attention_length] = jax.jit(
                lambda state: self.decode(state, attention_length),
                donate_argnums=(0,),
                **self.shardings.decode_shardings,
            )
        return self.precompile_dict["decode"][attention_length](inference_state)

    def single_rollout(self, state: InferenceState) -> tuple[Array, Array, dict[str, float]]:
        """
        Perform a single rollout for the given inference state and prompt tokens.
        Args:
            inference_state (InferenceState): The initial state of the inference, containing the current token, kv cache, and other necessary information.
            prompt_tokens (Array): The input prompt tokens to start the rollout. Shape: [decode_size, seq_len].
        Returns:
            Array: The output tokens generated during the rollout. Shape: [decode_size, total_seq_len].
            Array: The log probabilities of the generated tokens during the rollout. Shape: [decode_size, total_seq_len].
            dict[str, float]: Metrics collected during the rollout, such as time taken and tokens per second.
        """
        n_steps = 0
        start = time.perf_counter()
        with Tracker(timer=True) as t:
            while not jnp.all(state.stop_mask):
                state = self.decode_step(state)
                if (n_steps := n_steps + 1) % LOG_EVERY_N_STEPS == 0:
                    current_time = time.perf_counter() - start
                    logger.info(
                        f"Inferenced {n_steps} tokens | tps {self.decode_size * LOG_EVERY_N_STEPS / current_time:.2f} | sps {LOG_EVERY_N_STEPS / current_time:.2f}"
                    )
                    start = time.perf_counter()

        out_tokens, out_logprobs = jax.tree.map(
            lambda x: list(jax.device_get(x)), (state.out_tokens, state.out_logprobs)
        )

        return (
            out_tokens,
            out_logprobs,
            {"tps": n_steps * self.decode_size / t.data["time"], "sps": n_steps / t.data["time"]},
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

        @jax.jit
        def broadcast_to_decode_size(x: Array) -> Array:
            return jnp.broadcast_to(x, (self.decode_size,) + x.shape[1:])

        x_batch, seq_lens_batch = broadcast_to_decode_size(x), broadcast_to_decode_size(seq_lens)
        x_batch_sharded, seq_lens_sharded, params_sharded, key_sharded = self.put_batch_on_device(
            x_batch, seq_lens_batch, params, key
        )

        for step in range(self.config.group_size // self.decode_size):
            # NOTE:
            # we manually do prefill on each step instead of reusing
            # since then we do not need to keep a copy of a kv cache
            # prefill is very fast and so this save 2x memory
            prefill_key = jax.random.fold_in(key_sharded, step)
            state, prefill_metrics = self.prefill_step(x_batch_sharded, seq_lens_sharded, params_sharded, prefill_key)
            output_tokens, output_logprobs, decode_metrics = self.single_rollout(state)

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
        self, batch_tokens: Array, seq_lens: Array, key: Array, params: PyTree
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
        output = []
        metrics = []

        with Tracker(timer=True) as t:
            for i in range(B):
                key = jax.random.fold_in(key, i)
                batch_output, batch_metrics = self.rollout_group(
                    jax.lax.dynamic_index_in_dim(batch_tokens, i, axis=0),
                    jax.lax.dynamic_index_in_dim(seq_lens, i, axis=0),
                    params,
                    key,
                )
                output.append(batch_output)
                metrics.append(batch_metrics)

        metrics = jax.tree.map(lambda *x: sum(x) / len(x), *metrics) | {"total_time": t.data["time"]}
        metrics = {f"inference_metrics/{k}": v for k, v in metrics.items()}
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
        key = jax.random.fold_in(key, stax.get_rank())
        params = self.setup_parameters(params)
        return key, params

    def __call__(self, prompts: list[str], key: Array, params: PyTree) -> InferenceResults:
        """
        Perform inference for the given input prompts, random key, and model parameters.
        Args:
            prompts (list[str]): The list of input prompts to perform inference on.
            key (Array): The random key for any stochastic operations during inference.
            params (PyTree): The model parameters to use for inference.
            detokenize (bool): Whether to detokenize the output rollouts into strings. Default is False.
        Returns:
            InferenceResults: The results of the inference, containing the output rollouts, optionally the detokenized output strings, and any collected metrics.
        """
        key, params = self.multihost_prep(key, params)
        inp_tokens, seq_lens = self.tokenize(prompts)
        output_rollouts, metrics = self.batch_rollout(inp_tokens, seq_lens, key, params)
        output_strs = self.detokenizer(output_rollouts)
        sync_global_devices("inference_engine_sync")
        return InferenceResults(rollouts=output_rollouts, output_strs=output_strs, metrics=metrics)
