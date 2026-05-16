import threading
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import stax
from jax.experimental.multihost_utils import broadcast_one_to_all, sync_global_devices
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree
from stax import Tracker
from stax.logger import staxLogger as logger
from transformers import AutoTokenizer

from src.constants import INTERUPT_THINKING_PHARSE, SYSTEM_PROMPT, AsyncOptions
from src.data import InferenceRollout, Sample
from src.model import KVCache, Model

from .config import AsyncState, InferenceShardings, InferenceState, TrainerConfig
from .utils import _maybe_force_eos, _maybe_force_eot, naive_sample
from .worker import Worker

AXIS_NAME = "data"


def apply_prompt_template(text: str) -> str:
    return f"""Solve the following math problem step by step. Put your answer inside \\boxed{{}}.
{text}
Remember to put your answer inside \\boxed{{}}."""


def get_chat_template(system_prompt: bool, text: str) -> list[dict[str, str]]:
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": SYSTEM_PROMPT})
    chat.append({"role": "user", "content": apply_prompt_template(text)})
    return chat


class AsyncInferenceWorker(Worker):
    """
    Inference Engine for RLVR that performs static batching with group-based rollouts, supporting multi-host decoding
    """

    def __init__(self, trainer_config: TrainerConfig, async_options: AsyncOptions):
        self.trainer_config = trainer_config
        self.inference_config = trainer_config.loss_config.inference_config
        self.async_options = async_options

        self.model = Model(trainer_config.model_config)
        abstract_output = self.model.init_state(rng=jax.random.PRNGKey(0), tx=None, sharding=None, abstract=True)
        dummy_params = jax.tree.map(
            lambda x: jnp.zeros(x.shape, dtype=self.inference_config.params_dtype), abstract_output
        )

        self.validate_config()

        self.shardings: InferenceShardings = self.get_shardings(dummy_params)
        self.params = jax.device_put(dummy_params, self.shardings.replicate_sharding)
        jax.block_until_ready(self.params)

        self.async_state = AsyncState(MRUparams=jax.device_get(self.params), updated=False)

        self.monitor_thread = threading.Thread(
            target=self.monitor_weight_sync, args=(self.async_state, self.async_options), daemon=True
        )
        self.monitor_thread.start()

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

        self.prompt_id_offset = 0
        self.pid_to_sample = dict()
        self.global_rollouts: dict[int, InferenceRollout] = dict()

        self.weight_iteration = 0
        self.worker_rank = jax.process_index() - self.async_options.train_workers

        self.block_until_params_update()

    def monitor_weight_sync(self, async_state: AsyncState, async_options: AsyncOptions):
        def inference_sync_weights(params_cpu, async_options: AsyncOptions):
            sync_global_devices("weightSync")
            sync_global_devices("gathered")

            params_cpu = broadcast_one_to_all(params_cpu)
            logger.info("Inference workers received updated params", log_for_all=True)
            return params_cpu

        while True:
            async_options.weight_sync_queue.get()
            logger.info(
                f"Rank {jax.process_index()} received sync signal from train worker, syncing weights to latest parameters...",
                log_for_all=True,
            )
            new_params = inference_sync_weights(async_state.MRUparams, async_options)
            with async_state.update_lock:
                async_state.MRUparams = new_params
                async_state.updated = True
                async_state.weight_iteration += 1

    def _maybe_update_params(self):
        with self.async_state.update_lock:
            if self.async_state.updated:
                self.async_state.updated = False
                self.params = jax.tree.map(
                    lambda x, s: jax.device_put(x, s),
                    self.async_state.MRUparams,
                    self.shardings.params_sharding,
                )
                logger.info("Params updated on inference worker", log_for_all=True)
                self.weight_iteration = self.async_state.weight_iteration

    def block_until_params_update(self):
        logger.info("Waiting for initial parameters from training workers...", log_for_all=True)
        while True:
            with self.async_state.update_lock:
                if self.async_state.updated:
                    break
            time.sleep(0.1)
        self._maybe_update_params()

    def validate_config(self):
        """Validate the inference configuration to ensure it meets the requirements for the inference engine."""
        assert self.inference_config.n_replicas <= jax.local_device_count(), (
            f"Number of replicas {self.inference_config.n_replicas} must be less than or equal to number of devices {jax.device_count()}"
        )
        assert self.inference_config.max_seq_len <= self.model.sequence_len, (
            f"expected inference max seq len {self.inference_config.max_seq_len} to be less than model sequence length {self.model.sequence_len}"
        )
        assert self.inference_config.max_seq_len & (self.inference_config.max_seq_len - 1) == 0, (
            f"max_seq_len must be a power of 2, got {self.inference_config.max_seq_len}"
        )
        assert self.inference_config.initial_sequence_len & (self.inference_config.initial_sequence_len - 1) == 0, (
            f"initial_sequence_len must be a power of 2, got {self.inference_config.initial_sequence_len}"
        )

        assert self.inference_config.max_prefill_sequence_len <= self.inference_config.max_seq_len, (
            f"max_prefill_sequence_len {self.inference_config.max_prefill_sequence_len} must be less than or equal to max_seq_len {self.inference_config.max_seq_len}"
        )
        assert (
            self.inference_config.max_prefill_sequence_len & (self.inference_config.max_prefill_sequence_len - 1) == 0
        ), f"max_prefill_sequence_len must be a power of 2, got {self.inference_config.max_prefill_sequence_len}"

        assert self.inference_config._max_decode_prompts % (self.inference_config.n_replicas) == 0, (
            f"_max_decode_prompts {self.inference_config._max_decode_prompts} must be divisible by n_replicas {self.inference_config.n_replicas}"
        )

        assert (self.inference_config._max_decode_prompts * self.inference_config.group_size) % (
            self.inference_config._max_decode_batch_size
        ) == 0, (
            f"group_size * _max_decode_prompts {self.inference_config.group_size * self.inference_config._max_decode_prompts} must be divisible by _max_decode_batch_size {self.inference_config._max_decode_batch_size}"
        )

        assert self.inference_config._max_decode_batch_size % self.inference_config.n_replicas == 0, (
            f"_max_decode_batch_size {self.inference_config._max_decode_batch_size} must be divisible by n_replicas {self.inference_config.n_replicas}"
        )

        if self.inference_config.reasoning_budget is not None:
            answer_tokens = min(1024, self.inference_config.max_seq_len // 2)
            assert self.inference_config.reasoning_budget <= (self.inference_config.max_seq_len - answer_tokens), (
                f"Reasoning budget {self.inference_config.reasoning_budget} must be less than or equal to {self.inference_config.max_seq_len - answer_tokens} to account answer tokens"
            )

        if not self.inference_config.think_mode:
            assert self.inference_config.reasoning_budget is None, (
                "Reasoning budget should be None when think_mode is disabled"
            )

        if self.inference_config.top_k is not None:
            assert self.inference_config.top_k > 0, f"top_k must be positive, got {self.inference_config.top_k}"

        if self.inference_config.top_p is not None:
            assert 0.0 < self.inference_config.top_p <= 1.0, (
                f"top_p must be in the range (0, 1], got {self.inference_config.top_p}"
            )

    def get_shardings(self, params) -> InferenceShardings:
        """Get the shardings for the model parameters, kv cache, and inference state based on the configuration."""
        local_devices = np.array(jax.local_devices())
        mesh = jax.make_mesh(
            (self.inference_config.n_replicas,), (AXIS_NAME,), devices=local_devices[: self.inference_config.n_replicas]
        )

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

        replicate_state_sharding = jax.tree.map(lambda _x: replicate_sharding, state_sharding)

        params_sharding = jax.tree.map(lambda _p: replicate_sharding, params)

        # NOTE: use replicate sharding since we have to split aftewards into slices of
        # (1, ...) and so we will have to replicate
        prefill_shardings = {
            "in_shardings": (
                replicate_sharding,
                replicate_sharding,
                replicate_sharding,
                replicate_sharding,
                replicate_sharding,
            ),
            "out_shardings": replicate_state_sharding,
        }

        decode_any_sharding = {
            "in_shardings": (state_sharding, params_sharding, replicate_state_sharding, replicate_sharding),
            "out_shardings": (
                state_sharding,
                replicate_sharding,
                replicate_sharding,
                replicate_sharding,
                replicate_sharding,
            ),
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
        )

    def replicate_across_axis(self, value: Array) -> Array:
        return jax.device_put(value, self.shardings.replicate_sharding)

    def split_across_axis(self, value: Array) -> Array:
        return jax.device_put(value, self.shardings.split_sharding)

    def put_state_on_device(self, state: InferenceState) -> InferenceState:
        return jax.tree.map(lambda x, s: jax.device_put(x, s), state, self.shardings.state_sharding)

    def put_batch_on_device(self, batch: np.ndarray, seq_lens: np.ndarray, key: Array) -> tuple[Array, Array, Array]:
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
        return batch, seq_lens, key

    def compute_max_power_of_two(self, n: int, upper_bound: int) -> int:
        """Compute the maximum power of two less than or equal to n and upper_bound."""
        return min(1 << (n.bit_length()), upper_bound)

    def compute_max_padding_length(self, seq_lens: np.ndarray) -> int:
        """Compute the maximum padding length for the input batch based on the sequence lengths and the maximum sequence length."""
        return self.compute_max_power_of_two(max(seq_lens).item(), self.inference_config.max_seq_len)

    def tokenize(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        inputs: list[list[int]] = [
            self.tokenizer.apply_chat_template(
                get_chat_template(self.inference_config.system_prompt, text),
                add_generation_prompt=True,
                enable_thinking=self.inference_config.think_mode,
                tokenize=True,
            )
            for text in texts
        ]

        seq_lens = np.array([len(x) for x in inputs], dtype=np.int32)
        padding_length = max(self.compute_max_padding_length(seq_lens), self.inference_config.initial_sequence_len)
        inputs = [(padding_length - len(x)) * [self.tokenizer.pad_token_id] + x for x in inputs]
        tokens = np.array(inputs, dtype=np.int32)

        return tokens, seq_lens

    def cleanup_rollouts(self, pids: list[int]):
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

        for i in pids:
            new_rollouts = []
            new_logprobs = []
            for t, lps in zip(self.global_rollouts[i].rollout_tokens, self.global_rollouts[i].rollout_logprobs):
                t, lp = clean_sequence(t, lps)
                new_rollouts.append(t)
                new_logprobs.append(lp)
            self.global_rollouts[i].rollout_tokens = new_rollouts
            self.global_rollouts[i].rollout_logprobs = new_logprobs

    def detokenizer(self, pids: list[int]):
        for pid in pids:
            rollout_tokens = self.global_rollouts[pid].rollout_tokens
            rollout_strs = self.tokenizer.batch_decode(rollout_tokens, skip_special_tokens=False)
            self.global_rollouts[pid].rollout_strs = rollout_strs

    def prefill(
        self, input_tokens: Array, seq_lens: Array, params: PyTree, key: Array, prompt_id_offset: int
    ) -> InferenceState:
        logger.info(f"Compiling prefill for sequence length {input_tokens.shape[1]}", log_for_all=True)

        max_decode_prompts = self.inference_config._max_decode_prompts
        kv_cache_dtype = self.inference_config.kv_cache_dtype

        with jax.named_scope("prefill"):
            kv_cache = self.model.init_kv_cache(
                max_decode_prompts,
                length=self.max_attention_length,
                dtype=kv_cache_dtype,
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
                jnp.ones((max_decode_prompts, self.max_attention_length), dtype=jnp.int32) * self.tokenizer.pad_token_id
            )
            out_logprobs = jnp.zeros((max_decode_prompts, self.max_attention_length), dtype=jnp.float32)
            out_tokens = jax.lax.dynamic_update_slice_in_dim(out_tokens, input_tokens, 0, axis=1)
            out_logprobs = jax.lax.dynamic_update_slice_in_dim(
                out_logprobs, -jnp.inf * jnp.ones_like(input_tokens, dtype=jnp.float32), 0, axis=1
            )

        return InferenceState(
            next_token=input_tokens[:, -1:],
            seq_lens=seq_lens,
            kv_cache=out_cache,
            key=key,
            stop_mask=jnp.zeros((max_decode_prompts, 1), dtype=bool),
            end_of_think=jnp.zeros((max_decode_prompts, 1), dtype=bool),
            out_tokens=out_tokens,
            out_logprobs=out_logprobs,
            prompt_id=jnp.arange(input_tokens.shape[0])[:, None] + prompt_id_offset,
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
                input_tokens, seq_lens, params, key, self.prompt_id_offset
            )
            jax.tree.map(lambda x: x.block_until_ready(), out)
        return out, {"ttft": t.data["time"]}

    def decode(self, state: InferenceState, params: PyTree) -> InferenceState:
        logger.info(f"Compiling decode step for attention length {state.kv_cache[0].k.shape[1]}", log_for_all=True)
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
                temperature=self.inference_config.temperature,
                top_k=self.inference_config.top_k,
                top_p=self.inference_config.top_p,
            )

        with jax.named_scope("stop_masking"):
            end_of_think = state.end_of_think
            if self.inference_config.reasoning_budget is not None:
                next_token, next_log_prob, end_of_think = _maybe_force_eot(
                    next_token,
                    next_log_prob,
                    state.end_of_think,
                    state.seq_lens,
                    reasoning_budget=self.inference_config.reasoning_budget,
                    token_sequence=self.thinking_tokens,
                )

            next_token, next_log_prob, stop_mask = _maybe_force_eos(
                next_token,
                next_log_prob,
                state.stop_mask,
                state.seq_lens,
                max_seq_len=self.inference_config.max_seq_len,
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
            return KVCache(k=rolled_k, v=rolled_v, length=jnp.array(length))  # type: ignore

        return state.replace(  # type: ignore
            kv_cache=[roll_kv_cache(kv) for kv in state.kv_cache],
            out_tokens=jnp.roll(state.out_tokens, diff, axis=1),
            out_logprobs=jnp.roll(state.out_logprobs, diff, axis=1),
        )

    def sub(self, old_state, new_state, index):
        return InferenceState(
            next_token=old_state.next_token.at[index].set(new_state.next_token),
            kv_cache=[
                KVCache(
                    k=old_state.kv_cache[i].k.at[index].set(new_state.kv_cache[i].k),
                    v=old_state.kv_cache[i].v.at[index].set(new_state.kv_cache[i].v),
                    length=old_state.kv_cache[i].length.copy(),
                )
                for i in range(len(old_state.kv_cache))
            ],
            key=old_state.key,
            seq_lens=old_state.seq_lens.at[index].set(new_state.seq_lens),
            stop_mask=old_state.stop_mask.at[index].set(new_state.stop_mask),
            end_of_think=old_state.end_of_think.at[index].set(new_state.end_of_think),
            out_tokens=old_state.out_tokens.at[index].set(new_state.out_tokens),
            out_logprobs=old_state.out_logprobs.at[index].set(new_state.out_logprobs),
            prompt_id=old_state.prompt_id.at[index].set(new_state.prompt_id),
        )

    def shift_batch(self, state: InferenceState):
        max_seq = jnp.max(state.seq_lens)
        shift_back = -(state.kv_cache[0].length - max_seq)
        return state.replace(  # type: ignore
            kv_cache=[
                KVCache(
                    k=jnp.roll(kv.k, shift_back, axis=1),
                    v=jnp.roll(kv.v, shift_back, axis=1),
                    length=max_seq.copy(),  # type: ignore
                )
                for kv in state.kv_cache
            ],
            out_tokens=jnp.roll(state.out_tokens, shift_back, axis=1),
            out_logprobs=jnp.roll(state.out_logprobs, shift_back, axis=1),
        )

    def sub_batch(self, state: InferenceState, new_batch: InferenceState) -> InferenceState:
        logger.info("compiling sub batch")

        index = jnp.argmax(state.stop_mask[:, 0], keepdims=True)
        new_batch = self.roll_cache_to_length(new_batch, state.kv_cache[0].length)

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

            old_state = jax.lax.cond(
                sub_on_this_device[0], self.sub, lambda o, _n, _i: o, old_state, new_state, local_idx
            )

            return old_state

        state = _sub(state, new_batch, index)
        state = self.shift_batch(state)

        return state

    @partial(jax.jit, static_argnums=(0,))
    def create_initial_state(self, prompts: InferenceState, initial_ids: Array) -> InferenceState:
        @partial(
            jax.shard_map,
            mesh=self.shardings.mesh,
            in_specs=(P(), P("data")),
            out_specs=jax.tree.map(lambda x: x.spec, self.shardings.state_sharding),
        )
        def f(prompts, initial_ids):
            return jax.vmap(self.create_batch, in_axes=(None, 0))(prompts, initial_ids)

        stacked_state = f(prompts, initial_ids)
        stacked_state = stacked_state.replace(
            next_token=stacked_state.next_token[:, 0],
            kv_cache=[
                KVCache(k=k.k[:, 0], v=k.v[:, 0], length=jnp.asarray(k.length[0], dtype="int32"))  # type: ignore
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
        stacked_state_shifted = self.shift_batch(stacked_state)
        return stacked_state_shifted

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

    def continuous_batch(
        self, prompts: InferenceState, state: InferenceState | None
    ) -> tuple[InferenceState, dict[str, float]]:
        if self.precompile_dict["decode"].get("any_stop") is None:
            self.precompile_dict["decode"]["any_stop"] = jax.jit(
                self._decode_single_loop, donate_argnums=(0,), **self.shardings.decode_any_shardings
            )

        P = prompts.next_token.shape[0]

        global_ids = jax.device_get(prompts.prompt_id).flatten().tolist()
        local_to_global = {i: gid for i, gid in enumerate(global_ids)}

        prompt_queue: list[int] = [i for i in range(P) for _ in range(self.inference_config.group_size)]

        finished_tokens = []
        finished_logprobs = []
        finished_pids = []

        queued_steps = 0
        subbed_steps = 0

        with Tracker(timer=True) as t:
            if state is None:
                ids = [prompt_queue.pop() for _ in range(self.inference_config._max_decode_batch_size)]
                initial_ids: Array = jnp.array(ids, dtype=jnp.int32)
                initial_ids = jax.device_put(initial_ids, self.shardings.split_sharding)
                state = self.create_initial_state(prompts, initial_ids)
                for i in ids:
                    self.global_rollouts[local_to_global[i]].weight_iteration = self.weight_iteration

            while len(prompt_queue) > 0:
                next_index = prompt_queue.pop()

                (state, tokens, logprobs, prompt_ids, n_steps) = self.precompile_dict["decode"]["any_stop"](
                    state, self.params, prompts, next_index
                )

                finished_tokens.append(tokens)
                finished_logprobs.append(logprobs)
                finished_pids.append(prompt_ids)

                queued_steps += n_steps
                subbed_steps += 1
                self._maybe_update_params()
                self.global_rollouts[local_to_global[next_index]].weight_iteration = self.weight_iteration

            finished_tokens_cpu = list(map(lambda x: jax.device_get(x), finished_tokens))
            finished_logprobs_cpu = list(map(lambda x: jax.device_get(x), finished_logprobs))
            finished_pids_cpu = list(map(lambda x: jax.device_get(x), finished_pids))

            if isinstance(queued_steps, jnp.ndarray):
                queued_steps = queued_steps.item()

        finished_tokens_cpu = np.concat(finished_tokens_cpu, axis=0)
        finished_logprobs_cpu = np.concat(finished_logprobs_cpu, axis=0)
        finished_pids_cpu = np.concat(finished_pids_cpu, axis=0)

        for i in range(finished_tokens_cpu.shape[0]):
            pid = finished_pids_cpu[i].item()
            self.global_rollouts[pid].rollout_tokens.append(finished_tokens_cpu[i])
            self.global_rollouts[pid].rollout_logprobs.append(finished_logprobs_cpu[i])

        tokens_per_second = queued_steps * self.inference_config._max_decode_batch_size / t.data["time"]
        sequences_per_second = queued_steps / t.data["time"]
        decode_metrics = {
            "decode_steps": queued_steps,
            "decode_steps_subbed": subbed_steps,
            "decode_time": t.data["time"],
            "decode_tps": tokens_per_second,
            "decode_sps": sequences_per_second,
        }
        return state, decode_metrics

    def batch_rollout(
        self, batch_tokens: np.ndarray, seq_lens: np.ndarray, key: Array, prev_state: InferenceState | None
    ) -> tuple[InferenceState, dict[str, float]]:
        x_batch_sharded, seq_lens_sharded, key_sharded = self.put_batch_on_device(batch_tokens, seq_lens, key)
        prefill_state, prefill_metrics = self.prefill_step(x_batch_sharded, seq_lens_sharded, self.params, key_sharded)
        prev_state, decode_metrics = self.continuous_batch(prefill_state, prev_state)

        return prev_state, prefill_metrics | decode_metrics

    def get_samples(self):
        with Tracker(timer=True) as t1:
            samples: list[Sample] = []
            while len(samples) < self.inference_config._max_decode_prompts:
                samples.append(self.async_options.prompt_queue.get())

        with Tracker(timer=True) as t2:
            for i, sample in enumerate(samples):
                self.global_rollouts[self.prompt_id_offset + i] = InferenceRollout(
                    sample=sample, rollout_tokens=[], rollout_logprobs=[], rollout_strs=[], weight_iteration=-1
                )

            prompts = [sample.prompt for sample in samples]
            input_tokens, seq_lens = self.tokenize(prompts)

        return input_tokens, seq_lens, {"get_prompts_time": t1.data["time"], "tokenize_time": t2.data["time"]}

    def gather_rollouts(self):
        with Tracker(timer=True) as t:
            pid_ready_to_process: list[int] = []
            for k, v in self.global_rollouts.items():
                if len(v) == self.inference_config.group_size:
                    pid_ready_to_process.append(k)

            self.cleanup_rollouts(pid_ready_to_process)
            self.detokenizer(pid_ready_to_process)

            for pid in pid_ready_to_process:
                self.async_options.rollout_queue.put(self.global_rollouts[pid])
                del self.global_rollouts[pid]

            logger.info(
                f"Put {len(pid_ready_to_process)} rollouts into rollout queue, queue size: {self.async_options.rollout_queue.qsize()}, prompt queue size {self.async_options.prompt_queue.qsize()}",
                log_for_all=True,
            )

        return {"gather_time": t.data["time"], "ready_rollouts": len(pid_ready_to_process)}

    def inference(self):
        key = jax.device_get(jax.random.fold_in(jax.random.PRNGKey(1024), stax.get_rank()))
        prev_state = None

        while True:
            with Tracker(timer=True) as t:
                input_tokens, seq_lens, prompt_metrics = self.get_samples()
                key, gen_key = jax.random.split(key)
                prev_state, batch_metrics = self.batch_rollout(input_tokens, seq_lens, gen_key, prev_state)
                gather_metrics = self.gather_rollouts()

                self.prompt_id_offset += self.inference_config._max_decode_prompts

            metrics = (
                batch_metrics
                | gather_metrics
                | prompt_metrics
                | {"total_time": t.data["time"], "worker_id": self.worker_rank}
            )
            self.async_options.inference_metrics_queue.put(metrics)

    def start(self):
        with jax.set_mesh(self.shardings.mesh):
            self.inference()

    @property
    def max_attention_length(self) -> int:
        return min(self.inference_config.max_seq_len, self.model.sequence_len)
