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
PADDING_BUFFER = 3000


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
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": apply_system_prompt_template()})
    chat.append({"role": "user", "content": apply_prompt_template(text)})
    return chat


class InferenceEngine:
    """
    Inference Engine for RLVR that performs static batching with group-based rollouts, supporting multi-host decoding
    """

    def __init__(self, model: Model, params: PyTree, config: InferenceConfig):
        self.model = model
        self.config = config
        self.validate_config()

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
        assert self.config.initial_sequence_len & (self.config.initial_sequence_len - 1) == 0, (
            f"initial_sequence_len must be a power of 2, got {self.config.initial_sequence_len}"
        )

        assert self.config.max_prefill_sequence_len <= self.config.max_seq_len, (
            f"max_prefill_sequence_len {self.config.max_prefill_sequence_len} must be less than or equal to max_seq_len {self.config.max_seq_len}"
        )
        assert self.config.max_prefill_sequence_len & (self.config.max_prefill_sequence_len - 1) == 0, (
            f"max_prefill_sequence_len must be a power of 2, got {self.config.max_prefill_sequence_len}"
        )

        assert self.config._max_decode_prompts % (self.config.n_replicas) == 0, (
            f"_max_decode_prompts {self.config._max_decode_prompts} must be divisible by n_replicas {self.config.n_replicas}"
        )

        assert (self.config._max_decode_prompts * self.config.group_size) % (self.config._max_decode_batch_size) == 0, (
            f"group_size * _max_decode_prompts {self.config.group_size * self.config._max_decode_prompts} must be divisible by _max_decode_batch_size * n_replicas {self.config._max_decode_batch_size * self.config.n_replicas}"
        )

        assert self.config._max_decode_batch_size % self.config.n_replicas == 0, (
            f"_max_decode_batch_size {self.config._max_decode_batch_size} must be divisible by n_replicas {self.config.n_replicas}"
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
        mesh = jax.make_mesh((self.config.n_replicas,), (AXIS_NAME,), devices=local_devices[: self.config.n_replicas])

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
            prompt_id=split_sharding,  # type: ignore
        )

        params_sharding = jax.tree.map(lambda _p: replicate_sharding, params)

        # NOTE: use replicate sharding since we have to split aftewards into slices of
        # (1, ...) and so we will have to replicate
        prefill_shardings = {
            "in_shardings": (
                replicate_sharding,
                replicate_sharding,
                replicate_sharding,
                replicate_sharding,
            ),
            "out_shardings": state_sharding,
        }

        decode_any_sharding = {
            "in_shardings": (state_sharding, params_sharding, state_sharding, replicate_sharding),
            "out_shardings": (
                state_sharding,
                replicate_sharding,
                replicate_sharding,
                replicate_sharding,
                replicate_sharding,
            ),
        }

        decode_all_shardings = {
            "in_shardings": (
                state_sharding,
                params_sharding,
            ),
            "out_shardings": (state_sharding, replicate_sharding),
        }

        return InferenceShardings(
            mesh=mesh,
            split_sharding=split_sharding,
            replicate_sharding=replicate_sharding,
            kv_cache_sharding=kv_sharding,
            state_sharding=state_sharding,
            params_sharding=params_sharding,
            prefill_shardings=prefill_shardings,
            decode_any_shardings=decode_any_sharding,
            decode_all_shardings=decode_all_shardings,
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
        padding_length = max(self.compute_max_padding_length(seq_lens), self.config.initial_sequence_len)
        inputs = [(padding_length - len(x)) * [self.tokenizer.pad_token_id] + x for x in inputs]
        tokens = np.array(inputs, dtype=np.int32)

        if padding_length > PADDING_BUFFER:
            raise ValueError(
                f"Input sequence padded prompts (T={padding_length}) was greater than kv-cache length with padding, either implement roll cache or add additional buffer space"
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
                self.config._max_decode_prompts,
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
            out_tokens = (
                jnp.ones((self.config._max_decode_prompts, self.max_attention_length), dtype=jnp.int32)
                * self.tokenizer.pad_token_id
            )
            out_logprobs = jnp.zeros((self.config._max_decode_prompts, self.max_attention_length), dtype=jnp.float32)
            out_tokens = jax.lax.dynamic_update_slice_in_dim(out_tokens, input_tokens, 0, axis=1)
            out_logprobs = jax.lax.dynamic_update_slice_in_dim(
                out_logprobs, -jnp.inf * jnp.ones_like(input_tokens, dtype=jnp.float32), 0, axis=1
            )

        return InferenceState(
            next_token=input_tokens[:, -1:],
            seq_lens=seq_lens,
            kv_cache=out_cache,
            key=key,
            stop_mask=jnp.zeros((self.config._max_decode_prompts, 1), dtype=bool),
            end_of_think=jnp.zeros((self.config._max_decode_prompts, 1), dtype=bool),
            out_tokens=out_tokens,
            out_logprobs=out_logprobs,
            prompt_id=jnp.arange(input_tokens.shape[0])[:, None],
        )

    def prefill_step(
        self, input_tokens: Array, seq_lens: Array, params: PyTree, key: Array
    ) -> tuple[InferenceState, dict[str, float]]:
        if self.precompile_dict["prefill"].get((precompiled_length := input_tokens.shape[1])) is None:
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

        out_tokens = jax.lax.dynamic_update_index_in_dim(state.out_tokens, next_token, out_cache[0].length, axis=1)
        out_logprobs = jax.lax.dynamic_update_index_in_dim(
            state.out_logprobs, next_log_prob, out_cache[0].length, axis=1
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
            prompt_id=state.prompt_id,
        )

    def create_batch(self, prompts: InferenceState, index: int) -> InferenceState:
        def ds(x):
            return jax.lax.dynamic_slice_in_dim(x, index, 1, axis=0)

        return InferenceState(
            next_token=ds(prompts.next_token),
            kv_cache=[KVCache(k=ds(cache.k), v=ds(cache.v), length=cache.length) for cache in prompts.kv_cache],
            key=prompts.key,
            seq_lens=ds(prompts.seq_lens),
            stop_mask=ds(prompts.stop_mask),
            end_of_think=ds(prompts.end_of_think),
            out_tokens=ds(prompts.out_tokens),
            out_logprobs=ds(prompts.out_logprobs),
            prompt_id=ds(prompts.prompt_id),
        )

    def roll_cache_to_length(self, state: InferenceState, length: int) -> InferenceState:
        diff = length - state.kv_cache[0].length

        def roll_kv_cache(kv: KVCache) -> KVCache:
            rolled_k = jnp.roll(kv.k, diff, axis=1)
            rolled_v = jnp.roll(kv.v, diff, axis=1)
            return KVCache(k=rolled_k, v=rolled_v, length=jnp.array(length))

        return state.replace(
            kv_cache=[roll_kv_cache(kv) for kv in state.kv_cache],
            out_tokens=jnp.roll(state.out_tokens, diff, axis=1),
            out_logprobs=jnp.roll(state.out_logprobs, diff, axis=1),
        )

    def sub_batch(self, state: InferenceState, new_batch: InferenceState) -> InferenceState:
        logger.info("compiling sub batch")

        index = jnp.argmax(state.stop_mask[:, 0], keepdims=True)
        current_length = state.kv_cache[0].length
        new_batch = self.roll_cache_to_length(new_batch, current_length)

        def sub(old_state, new_state, index):
            return InferenceState(
                next_token=old_state.next_token.at[index].set(new_state.next_token),
                kv_cache=[
                    KVCache(
                        k=old_state.kv_cache[i].k.at[index].set(new_state.kv_cache[i].k),
                        v=old_state.kv_cache[i].v.at[index].set(new_state.kv_cache[i].v),
                        length=current_length.copy(),
                    )
                    for i in range(len(state.kv_cache))
                ],
                key=old_state.key,
                seq_lens=old_state.seq_lens.at[index].set(new_state.seq_lens),
                stop_mask=old_state.stop_mask.at[index].set(new_state.stop_mask),
                end_of_think=old_state.end_of_think.at[index].set(new_state.end_of_think),
                out_tokens=old_state.out_tokens.at[index].set(new_state.out_tokens),
                out_logprobs=old_state.out_logprobs.at[index].set(new_state.out_logprobs),
                prompt_id=old_state.prompt_id.at[index].set(new_state.prompt_id),
            )

        @partial(
            jax.shard_map,
            mesh=self.shardings.mesh,
            in_specs=(
                jax.tree.map(lambda x: x.spec, self.shardings.state_sharding),
                jax.tree.map(lambda x: P(), self.shardings.state_sharding),
                P(),
            ),
            out_specs=(jax.tree.map(lambda x: x.spec, self.shardings.state_sharding)),
        )
        def _sub(old_state: InferenceState, new_state: InferenceState, index: Array):
            B = old_state.next_token.shape[0]
            device_id = jax.lax.axis_index(AXIS_NAME)

            start_idx = B * device_id
            end_idx = B * (device_id + 1)

            local_idx = index - start_idx

            sub_on_this_device = (index >= start_idx) & (index < end_idx)

            state = jax.lax.cond(
                sub_on_this_device[0], sub, lambda old_state, _n, _i: old_state, old_state, new_state, local_idx
            )

            return state

        state = _sub(state, new_batch, index)

        max_seq = jnp.max(state.seq_lens)
        shift_back = -(current_length - max_seq)
        state = state.replace(
            kv_cache=[
                KVCache(
                    k=jnp.roll(kv.k, shift_back, axis=1),
                    v=jnp.roll(kv.v, shift_back, axis=1),
                    length=max_seq.copy(),
                )
                for kv in state.kv_cache
            ],
            out_tokens=jnp.roll(state.out_tokens, shift_back, axis=1),
            out_logprobs=jnp.roll(state.out_logprobs, shift_back, axis=1),
        )

        return state

    @partial(jax.jit, static_argnums=(0,))
    def create_initial_state(self, prompts: InferenceState, initial_ids: Array) -> InferenceState:
        @partial(
            jax.shard_map,
            mesh=self.shardings.mesh,
            in_specs=(jax.tree.map(lambda _x: P(), self.shardings.state_sharding), P("data")),
            out_specs=jax.tree.map(lambda x: x.spec, self.shardings.state_sharding),
        )
        def f(prompts, initial_ids):
            return jax.vmap(self.create_batch, in_axes=(None, 0))(prompts, initial_ids)

        stacked_state = f(prompts, initial_ids)
        stacked_state = stacked_state.replace(
            next_token=stacked_state.next_token[:, 0],
            kv_cache=[
                KVCache(k=k.k[:, 0], v=k.v[:, 0], length=jnp.asarray(k.length[0], dtype="int32"))
                for k in stacked_state.kv_cache
            ],
            key=stacked_state.key[0],
            seq_lens=stacked_state.seq_lens[:, 0],
            stop_mask=stacked_state.stop_mask[:, 0],
            end_of_think=stacked_state.end_of_think[:, 0],
            out_tokens=stacked_state.out_tokens[:, 0],
            out_logprobs=stacked_state.out_logprobs[:, 0],
            prompt_id=stacked_state.prompt_id[:, 0],
        )

        return stacked_state

    def _decode_loop(self, state: InferenceState, params: PyTree) -> tuple[InferenceState, int]:
        before_length = state.kv_cache[0].length
        state = jax.lax.while_loop(
            lambda state: ~jnp.all(state.stop_mask),
            lambda state: self.decode(state, params),
            state,
        )
        after_length = state.kv_cache[0].length
        return state, (after_length - before_length)

    def _decode_single_loop(
        self, state: InferenceState, params: PyTree, prompts: InferenceState, next_index: int
    ) -> tuple[InferenceState, Array, Array, Array, int]:
        with jax.named_scope("single_decode_loop"):
            before_length = state.kv_cache[0].length
            state = jax.lax.while_loop(
                lambda state: ~jnp.any(state.stop_mask),
                lambda state: self.decode(state, params),
                state,
            )
            after_length = state.kv_cache[0].length

            index = jnp.argmax(state.stop_mask[:, 0], keepdims=True)
            tokens, logprobs, prompt_ids = (
                state.out_tokens[index],
                state.out_logprobs[index],
                state.prompt_id[index],
            )

        with jax.named_scope("sub_next_batch"):
            next_batch = self.create_batch(prompts, next_index)
            state = self.sub_batch(state, next_batch)

        return state, tokens, logprobs, prompt_ids, (after_length - before_length)

    def continuous_batch(self, prompts: InferenceState, params: PyTree):
        if self.precompile_dict["decode"].get("any_stop") is None:
            self.precompile_dict["decode"]["any_stop"] = jax.jit(
                self._decode_single_loop, donate_argnums=(0,), **self.shardings.decode_any_shardings
            )

        if self.precompile_dict["decode"].get("all_stop") is None:
            self.precompile_dict["decode"]["all_stop"] = jax.jit(
                self._decode_loop, donate_argnums=(0,), **self.shardings.decode_all_shardings
            )

        P = prompts.next_token.shape[0]
        prompt_queue: list[int] = [i for i in range(P) for _ in range(self.config.group_size)]

        grouped_tokens: list[list[np.ndarray]] = [[] for _ in range(P)]
        grouped_logprobs: list[list[np.ndarray]] = [[] for _ in range(P)]

        finished_tokens = []
        finished_logprobs = []
        finished_prompt_ids = []
        decode_steps = 0
        queued_steps = 0
        subbed_steps = 0

        with Tracker(timer=True) as t:
            intial_ids = jnp.array(
                [prompt_queue.pop() for _ in range(self.config._max_decode_batch_size)], dtype=jnp.int32
            )
            intial_ids = jax.device_put(intial_ids, self.shardings.split_sharding)
            state = self.create_initial_state(prompts, intial_ids)
            while len(prompt_queue) > 0:
                next_index = prompt_queue.pop()

                state, tokens, logprobs, prompt_ids, n_steps = self.precompile_dict["decode"]["any_stop"](
                    state, params, prompts, next_index
                )

                finished_tokens.append(tokens)
                finished_logprobs.append(logprobs)
                finished_prompt_ids.append(prompt_ids)

                queued_steps += n_steps
                subbed_steps += 1

            state, final_steps = self.precompile_dict["decode"]["all_stop"](state, params)

            tokens, logprobs, prompt_ids = (
                state.out_tokens,
                state.out_logprobs,
                state.prompt_id,
            )
            finished_tokens.append(tokens)
            finished_logprobs.append(logprobs)
            finished_prompt_ids.append(prompt_ids)

            finished_tokens_cpu = list(map(lambda x: jax.device_get(x), finished_tokens))
            finished_logprobs_cpu = list(map(lambda x: jax.device_get(x), finished_logprobs))
            finished_prompt_ids_cpu = list(map(lambda x: jax.device_get(x), finished_prompt_ids))

            if isinstance(final_steps, jnp.ndarray):
                final_steps = final_steps.item()
            if isinstance(queued_steps, jnp.ndarray):
                queued_steps = queued_steps.item()

        finished_tokens_cpu = np.concat(finished_tokens_cpu, axis=0)
        finished_logprobs_cpu = np.concat(finished_logprobs_cpu, axis=0)
        finished_prompt_ids_cpu = np.concat(finished_prompt_ids_cpu, axis=0)

        for i in range(finished_tokens_cpu.shape[0]):
            pid = finished_prompt_ids_cpu[i].item()
            grouped_tokens[pid].append(finished_tokens_cpu[i])
            grouped_logprobs[pid].append(finished_logprobs_cpu[i])

        rollouts = [InferenceRollout(rollouts=t, logprobs=lp) for t, lp in zip(grouped_tokens, grouped_logprobs)]

        decode_steps = queued_steps + final_steps
        decode_tokens = decode_steps * self.config._max_decode_batch_size * jax.process_count()

        decode_metrics = {
            "decode_steps": decode_steps,
            "queued_steps": queued_steps,
            "subbed_steps": subbed_steps,
            "final_steps": final_steps,
            "decode_time": t.data["time"],
            "decode_tokens": decode_tokens,
            "decode_tps": decode_tokens / t.data["time"],
            "decode_sps": decode_steps / t.data["time"],
        }
        return rollouts, decode_metrics

    def batch_rollout(
        self, batch_tokens: np.ndarray, seq_lens: np.ndarray, key: Array, params: PyTree
    ) -> tuple[list[InferenceRollout], PyTree]:
        B, _ = batch_tokens.shape
        output: list[InferenceRollout] = []
        metrics: list[dict[str, float]] = []

        assert (B := batch_tokens.shape[0]) % self.config._max_decode_prompts == 0, (
            f"Batch size {B} must be divisible by _max_decode_prompts {self.config._max_decode_prompts} for static batching"
        )

        x_batch_sharded, seq_lens_sharded, params_sharded, key_sharded = self.put_batch_on_device(
            batch_tokens, seq_lens, params, key
        )

        n_steps = B // self.config._max_decode_prompts

        for i in range(n_steps):
            batch_key = jax.random.fold_in(key, i)

            start = i * self.config._max_decode_prompts
            current_batch = jax.lax.dynamic_slice_in_dim(
                x_batch_sharded, start, self.config._max_decode_prompts, axis=0
            )
            current_seq_lens = jax.lax.dynamic_slice_in_dim(
                seq_lens_sharded, start, self.config._max_decode_prompts, axis=0
            )

            prefill_state, prefill_metrics = self.prefill_step(
                current_batch, current_seq_lens, params_sharded, batch_key
            )

            batch_output, batch_metrics = self.continuous_batch(
                prefill_state,
                params_sharded,
            )

            output.extend(batch_output)
            metrics.append(batch_metrics | prefill_metrics)

            del prefill_state

        metrics: dict[str, float] = jax.tree.map(lambda *x: sum(x) / len(x), *metrics)
        output = self.cleanup_rollouts(output)

        return output, metrics

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
