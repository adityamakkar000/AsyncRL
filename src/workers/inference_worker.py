import contextlib
import threading
import time
from queue import Empty

import jax
import jax.numpy as jnp
import numpy as np
import stax
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree
from stax import Tracker
from stax.logger import staxLogger as logger

from src.constants import INTERUPT_THINKING_PHARSE, TIMEOUT, AsyncOptions
from src.data import (
    InferenceRollout,
    Sample,
    apply_chat_template,
    decode_tokens,
    load_tokenizer,
    resolve_pad_eos,
)
from src.model import KVCache, Model

from .config import AsyncState, InferenceConfig, InferenceState, TrainerConfig
from .rdma_transfer import RDMATransferServer
from .utils import (
    _maybe_force_eos,
    _maybe_force_eot,
    naive_sample,
)
from .worker import Worker

GLOBAL_AXIS = "global"
TP_AXIS = "tp"
COMPILE_LOCK = threading.Lock()


class SingleInferenceThread:
    """Runs an independent continuous-batching decode loop pinned to one local device."""

    def __init__(
        self,
        config: InferenceConfig,
        model: Model,
        async_state: AsyncState,
        async_options: AsyncOptions,
        worker_id: int,
        inference_rank: int,
        devices: np.ndarray,
        *,
        eval_group_size: int | None = None,
    ):
        self.config = config

        self.model = model
        self.tokenizer = load_tokenizer(self.model.config.hf_model_name)
        self.pad_token, self.eos_token = resolve_pad_eos(self.tokenizer)
        self.thinking_tokens = (
            self.tokenizer(INTERUPT_THINKING_PHARSE, add_special_tokens=False, return_tensors="np")
            .input_ids[0]
            .tolist()
        )

        self.async_options = async_options
        self.async_state = async_state
        self.weight_iteration = 0

        self.worker_id = worker_id
        self.inference_rank = inference_rank
        self.devices = devices
        self.tp = devices.size
        self.global_worker_id = (jax.local_device_count() // self.tp) * inference_rank + worker_id
        self.mesh = jax.sharding.Mesh(devices, axis_names=(TP_AXIS,), axis_types=(jax.sharding.AxisType.Explicit,))
        self.sharding = jax.NamedSharding(self.mesh, P())
        self.kv_cache_sharding = KVCache(k=self.sharding, v=self.sharding, length=self.sharding)  # type: ignore

        self.prefill_fns = {}
        self.decode_fn = None

        self.prompt_id_offset = 0
        self.global_rollouts: dict[int, InferenceRollout] = {}

        self.eval_group_size = eval_group_size

        self.block_until_params_update()

    def start(self, key: Array):
        key = jax.device_put(key, self.sharding)
        prev_state = None

        while True:
            with Tracker(timer=True) as t:
                input_tokens, seq_lens, prompt_metrics = self.get_samples()
                key, gen_key = jax.random.split(key)
                prev_state, batch_metrics = self.batch_rollout(input_tokens, seq_lens, gen_key, prev_state)
                gather_metrics = self.gather_rollouts()

                self.prompt_id_offset += self.max_prefill_prompts

            metrics = (
                batch_metrics
                | gather_metrics
                | prompt_metrics
                | {"total_time": t.data["time"], "worker_id": self.global_worker_id}
            )
            self.async_options.queues.inference_metrics_queue.put(metrics)

    def get_samples(self):
        with Tracker(timer=True) as t1:
            samples: list[tuple[Sample, bool]] = []
            while len(samples) < self.max_prefill_prompts:
                try:
                    samples.append((self.async_options.queues.eval_prompt_queue.get_nowait(), True))
                except Empty:
                    break

            n_eval = len(samples)
            while len(samples) < self.max_prefill_prompts:
                samples.append((self.async_options.queues.prompt_queue.get(timeout=TIMEOUT), False))

        with Tracker(timer=True) as t2:
            for i, (sample, is_eval) in enumerate(samples):
                self.global_rollouts[self.prompt_id_offset + i] = InferenceRollout(
                    sample=sample,
                    rollout_tokens=[],
                    rollout_logprobs=[],
                    rollout_strs=[],
                    weight_iteration=[],
                    n_rollouts=(self.eval_group_size or self.config.group_size) if is_eval else self.config.group_size,
                    is_eval=is_eval,
                )

            prompts = [sample.prompt for sample, _ in samples]
            input_tokens, seq_lens = self.tokenize(prompts)

        metrics = {"get_prompts_time": t1.data["time"], "tokenize_time": t2.data["time"], "eval_prompts": n_eval}
        return input_tokens, seq_lens, metrics

    def batch_rollout(
        self, batch_tokens: np.ndarray, seq_lens: np.ndarray, key: Array, prev_state: InferenceState | None
    ) -> tuple[InferenceState, dict[str, float]]:
        x_batch_sharded = jax.device_put(batch_tokens, self.sharding)
        seq_lens_sharded = jax.device_put(seq_lens, self.sharding)

        prefill_state, prefill_metrics = self.prefill_step(x_batch_sharded, seq_lens_sharded, self.params, key)
        prev_state, decode_metrics = self.continuous_batch(prefill_state, prev_state)

        return prev_state, prefill_metrics | decode_metrics

    def prefill_step(
        self, input_tokens: Array, seq_lens: Array, params: PyTree, key: Array
    ) -> tuple[InferenceState, dict[str, float]]:
        first_compile = self.prefill_fns.get(precompiled_length := input_tokens.shape[1]) is None
        if first_compile:
            self.prefill_fns[precompiled_length] = jax.jit(self.prefill)

        with Tracker(timer=True) as t:
            with COMPILE_LOCK if first_compile else contextlib.nullcontext():
                out: InferenceState = self.prefill_fns[precompiled_length](
                    input_tokens, seq_lens, params, key, self.prompt_id_offset
                )
            jax.tree.map(lambda x: x.block_until_ready(), out)
        return out, {"ttft": t.data["time"]}

    def prefill(
        self, input_tokens: Array, seq_lens: Array, params: PyTree, key: Array, prompt_id_offset: int
    ) -> InferenceState:
        logger.info(f"Compiling prefill for sequence length {input_tokens.shape[1]}", log_for_all=True)

        max_prefill_prompts = self.max_prefill_prompts
        kv_cache_dtype = self.config.kv_cache_dtype

        with jax.named_scope("prefill"):
            kv_cache = self.model.init_kv_cache(
                max_prefill_prompts,
                length=self.max_attention_length,
                dtype=kv_cache_dtype,
                sharding=self.kv_cache_sharding,
            )
            _logits, out_cache = self.model.apply(
                params, x=input_tokens[:, :-1], sequence_lens=seq_lens - 1, kv_cache=kv_cache
            )
            out_tokens = jnp.ones((max_prefill_prompts, self.max_attention_length), dtype=jnp.int32) * self.pad_token
            out_logprobs = jnp.zeros((max_prefill_prompts, self.max_attention_length), dtype=jnp.float32)
            out_tokens = jax.lax.dynamic_update_slice_in_dim(out_tokens, input_tokens, 0, axis=1)
            out_logprobs = jax.lax.dynamic_update_slice_in_dim(
                out_logprobs, -jnp.inf * jnp.ones_like(input_tokens, dtype=jnp.float32), 0, axis=1
            )

        return InferenceState(
            next_token=input_tokens[:, -1:],
            seq_lens=seq_lens,
            kv_cache=out_cache,
            key=key,
            stop_mask=jnp.zeros((max_prefill_prompts, 1), dtype=bool),
            end_of_think=jnp.zeros((max_prefill_prompts, 1), dtype=bool),
            out_tokens=out_tokens,
            out_logprobs=out_logprobs,
            prompt_id=jnp.arange(input_tokens.shape[0])[:, None] + prompt_id_offset,
        )

    def continuous_batch(
        self, prompts: InferenceState, state: InferenceState | None
    ) -> tuple[InferenceState, dict[str, float]]:
        first_compile = self.decode_fn is None
        if first_compile:
            self.decode_fn = jax.jit(self._decode_single_loop, donate_argnums=(0,))

        n_prompts = prompts.next_token.shape[0]

        global_ids = jax.device_get(prompts.prompt_id).flatten().tolist()
        local_to_global = {i: gid for i, gid in enumerate(global_ids)}

        prompt_queue: list[int] = [
            i for i in range(n_prompts) for _ in range(self.global_rollouts[local_to_global[i]].n_rollouts)
        ]
        finished = {"tokens": [], "logprobs": [], "prompt_ids": []}

        queued_steps = jnp.asarray(0, dtype=jnp.int32)
        subbed_steps = 0

        with Tracker(timer=True) as t:
            if state is None:
                ids = [prompt_queue.pop() for _ in range(self.config.max_decode_batch_size)]
                initial_ids: Array = jnp.array(ids, dtype=jnp.int32)
                initial_ids = jax.device_put(initial_ids, self.sharding)
                state = self.create_initial_state(prompts, initial_ids)
                for i in ids:
                    self.global_rollouts[local_to_global[i]].weight_iteration.append(self.weight_iteration)

            while len(prompt_queue) > 0:
                next_index = prompt_queue.pop()

                with COMPILE_LOCK if first_compile else contextlib.nullcontext():
                    (state, tokens, logprobs, prompt_ids, n_steps) = self.decode_fn(
                        state, self.params, prompts, next_index
                    )
                    if first_compile:
                        jax.block_until_ready(state)
                        first_compile = False

                finished["tokens"].append(tokens)
                finished["logprobs"].append(logprobs)
                finished["prompt_ids"].append(prompt_ids)

                queued_steps += n_steps
                subbed_steps += 1
                self._maybe_update_params()
                self.global_rollouts[local_to_global[next_index]].weight_iteration.append(self.weight_iteration)

            gathered = {k: jax.device_get(jnp.concatenate(v, axis=0)) for k, v in finished.items() if v}
            queued_steps = queued_steps.item()

        for i in range(gathered["tokens"].shape[0] if gathered else 0):
            pid = gathered["prompt_ids"][i].item()
            self.global_rollouts[pid].rollout_tokens.append(gathered["tokens"][i])
            self.global_rollouts[pid].rollout_logprobs.append(gathered["logprobs"][i])

        tokens_per_second = queued_steps * self.config.max_decode_batch_size / t.data["time"]
        sequences_per_second = queued_steps / t.data["time"]
        decode_metrics = {
            "decode_steps": queued_steps,
            "decode_steps_subbed": subbed_steps,
            "decode_time": t.data["time"],
            "decode_tps": tokens_per_second,
            "decode_sps": sequences_per_second,
        }
        return state, decode_metrics

    def create_initial_state(self, prompts: InferenceState, initial_ids: Array) -> InferenceState:
        def f(prompts, initial_ids):
            return jax.vmap(self.create_batch, in_axes=(None, 0))(prompts, initial_ids)

        stacked_state = f(prompts, initial_ids)
        stacked_state = stacked_state.replace(
            next_token=stacked_state.next_token[:, 0],
            kv_cache=[
                KVCache(k=k.k[:, 0], v=k.v[:, 0], length=jnp.asarray(k.length[0], dtype=jnp.int32))  # type: ignore
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
        stacked_state_shifted = stacked_state.shift_batch()
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
                eos_token_id=self.eos_token,
            )

        out_tokens = jax.lax.dynamic_update_index_in_dim(state.out_tokens, next_token, out_cache[0].length, axis=1)
        out_logprobs = jax.lax.dynamic_update_index_in_dim(
            state.out_logprobs, next_log_prob, out_cache[0].length, axis=1
        )

        return InferenceState(
            next_token=next_token,
            kv_cache=out_cache,
            key=key,
            seq_lens=state.seq_lens + 1,
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

    def sub_batch(self, state: InferenceState, new_batch: InferenceState) -> InferenceState:
        logger.info("compiling sub batch")

        index = jnp.argmax(state.stop_mask[:, 0], keepdims=True)
        kv_index = jnp.maximum(jnp.max(state.seq_lens), jnp.max(new_batch.seq_lens)) - 1
        new_batch = new_batch.roll(kv_index)
        state = state.roll(kv_index)
        state = state.sub(new_batch, index).shift_batch()
        return state

    def gather_rollouts(self):
        with Tracker(timer=True) as t:
            pid_ready_to_process: list[int] = []
            for k, v in self.global_rollouts.items():
                if len(v) == v.n_rollouts:
                    pid_ready_to_process.append(k)

            self.cleanup_rollouts(pid_ready_to_process)
            self.detokenizer(pid_ready_to_process)

            n_eval = 0
            for pid in pid_ready_to_process:
                rollout = self.global_rollouts[pid]
                if rollout.is_eval:
                    self.async_options.queues.eval_rollout_queue.put(rollout)
                    n_eval += 1
                else:
                    self.async_options.queues.rollout_queue.put(rollout)
                del self.global_rollouts[pid]

            logger.info(
                f"Put {len(pid_ready_to_process)} rollouts into rollout queue, queue size: {self.async_options.queues.rollout_queue.qsize()}, prompt queue size {self.async_options.queues.prompt_queue.qsize()}",
                log_for_all=True,
            )

        return {
            "gather_time": t.data["time"],
            "ready_rollouts": len(pid_ready_to_process),
            "ready_eval_rollouts": n_eval,
        }

    def cleanup_rollouts(self, pids: list[int]):
        def clean_sequence(tokens: np.ndarray, logprobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            non_pad_index = np.argmax(tokens != self.pad_token)
            tokens_not_padded = tokens[non_pad_index:]
            logprobs_not_padded = logprobs[non_pad_index:]

            eos_idx = np.where((tokens_not_padded != self.eos_token) & (tokens_not_padded != self.pad_token))[0][-1] + 1
            if eos_idx == tokens_not_padded.shape[0] or tokens_not_padded[eos_idx] != self.eos_token:
                raise ValueError(
                    "No eos token found in rollout.\nToken sequence: "
                    + str(tokens_not_padded)
                    + "\nDetokenized string: "
                    + decode_tokens(self.tokenizer, tokens_not_padded)
                )

            return tokens_not_padded[: eos_idx + 1], logprobs_not_padded[: eos_idx + 1]

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
            rollout_tokens = [t.tolist() for t in self.global_rollouts[pid].rollout_tokens]
            rollout_strs = self.tokenizer.batch_decode(rollout_tokens, skip_special_tokens=False)
            self.global_rollouts[pid].rollout_strs = rollout_strs

    def _maybe_update_params(self):
        with self.async_state.read_write_lock:
            if self.weight_iteration != self.async_state.weight_iteration:
                self.params = jax.tree.map(
                    lambda x: jax.device_put(x.addressable_shards[self.worker_id * self.tp].data, self.sharding),
                    self.async_state.MRUparams,
                )
                logger.info(f"Params updated on inference worker {self.global_worker_id}", log_for_all=True)
                self.weight_iteration = self.async_state.weight_iteration

    def block_until_params_update(self):
        logger.info("Waiting for initial parameters from training workers...", log_for_all=True)
        current_id = self.weight_iteration
        while True:
            self._maybe_update_params()
            if current_id < self.weight_iteration:
                break
            time.sleep(0.2)

    def tokenize(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        inputs: list[list[int]] = [
            apply_chat_template(
                self.tokenizer,
                self.config.system_prompt,
                text,
                enable_thinking=self.config.think_mode,
            )
            for text in texts
        ]

        seq_lens = np.array([len(x) for x in inputs], dtype=np.int32)
        padding_length = max(self.compute_max_padding_length(seq_lens), self.config.initial_sequence_len)
        inputs = [(padding_length - len(x)) * [self.pad_token] + x for x in inputs]
        tokens = np.array(inputs, dtype=np.int32)

        return tokens, seq_lens

    def compute_max_padding_length(self, seq_lens: np.ndarray) -> int:
        return self.compute_max_power_of_two(max(seq_lens).item(), self.max_attention_length)

    def compute_max_power_of_two(self, n: int, upper_bound: int) -> int:
        return min(1 << (n.bit_length()), upper_bound)

    @property
    def max_prefill_prompts(self) -> int:
        return 2 * self.config.max_decode_batch_size

    @property
    def max_attention_length(self) -> int:
        return min(self.config.max_seq_len, self.model.sequence_len)


class AsyncInferenceWorker(Worker):
    """
    Inference Engine for RLVR that performs static batching with group-based rollouts, supporting multi-host decoding
    """

    def __init__(self, trainer_config: TrainerConfig, async_options: AsyncOptions):
        self.trainer_config = trainer_config
        self.inference_config = trainer_config.loss_config.inference_config
        self.eval_config = trainer_config.eval_config
        self.async_options = async_options

        self.model = Model(trainer_config.model_config)

        abstract_output = self.model.init_state(rng=jax.random.PRNGKey(0), tx=None, sharding=None, abstract=True)
        dummy_params = jax.tree.map(
            lambda x: jnp.zeros(x.shape, dtype=self.inference_config.params_dtype), abstract_output
        )
        self.validate_config()

        self.mesh = jax.sharding.Mesh(
            np.array(jax.local_devices()), axis_names=(GLOBAL_AXIS,), axis_types=(jax.sharding.AxisType.Explicit,)
        )

        self.sharding = jax.NamedSharding(self.mesh, P())
        self.params = jax.device_put(dummy_params, self.sharding)

        self.async_state = AsyncState(MRUparams=self.params)

        self.worker_rank = stax.get_rank() - self.async_options.train_workers
        self.transfer_server = RDMATransferServer(train_workers=self.async_options.train_workers)

        self.monitor_thread = threading.Thread(
            target=self.monitor_weight_sync, args=(self.async_state, self.async_options), daemon=True
        )
        self.monitor_thread.start()

        device_groups = np.array(jax.local_devices()).reshape(-1, self.inference_config.tp)

        workers = [
            SingleInferenceThread(
                self.inference_config,
                self.model,
                self.async_state,
                async_options,
                i,
                self.worker_rank,
                device_groups[i],
                eval_group_size=self.eval_config.group_size,
            )
            for i in range(device_groups.shape[0])
        ]

        init_keys = jax.random.split(jax.random.PRNGKey(1024), len(workers))
        self.thread_workers = [threading.Thread(target=w.start, args=(init_keys[i],)) for i, w in enumerate(workers)]

    def start(self):
        for t in self.thread_workers:
            t.start()
        for t in self.thread_workers:
            t.join()

    def monitor_weight_sync(self, async_state: AsyncState, async_options: AsyncOptions):
        while True:
            _ = async_options.queues.weight_sync_queue.get()

            logger.info(
                f"[weight_sync] Rank {stax.get_rank()} received sync signal from train worker, syncing weights to latest parameters...",
                log_for_all=True,
            )

            new_params = self.transfer_server.transfer(async_state.MRUparams)

            with async_state.read_write_lock:
                async_state.MRUparams = new_params
                async_state.weight_iteration += 1

    def validate_config(self):
        """Validate the inference configuration to ensure it meets the requirements for the inference engine."""
        assert jax.local_device_count() % self.inference_config.tp == 0, (
            f"tp {self.inference_config.tp} must divide local device count {jax.local_device_count()}"
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

        if self.inference_config.tp > 1:
            raise NotImplementedError("tp >1 is not supported right now")
