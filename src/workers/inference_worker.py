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

from src.constants import INTERUPT_THINKING_PHARSE
from src.data import (
    InferenceRollout,
    Sample,
    apply_chat_template,
    decode_tokens,
    load_tokenizer,
    resolve_pad_eos,
)
from src.model import KVCache, Model

from .config import AsyncOptions, AsyncState, InferenceConfig, InferenceState, TrainerConfig
from .constants import TIMEOUT
from .rdma_transfer import RDMATransferServer
from .utils import (
    _maybe_force_eos,
    _maybe_force_eot,
    naive_sample,
)
from .worker import Worker

GLOBAL_AXIS = "global"
TP_AXIS = "tp"


class SingleInferenceReplica:
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
        self.initial_state_fn = None

        self.prompt_id_offset = 0
        self.global_rollouts: dict[int, InferenceRollout] = {}

        self.eval_group_size = eval_group_size

    def start(self, key: Array):
        key = jax.device_put(key, self.sharding)
        prev_state = None
        self.block_until_params_update()
        self.warmup()

        while True:
            with Tracker(timer=True) as t:
                input_tokens, seq_lens, prompt_metrics = self.get_samples()
                prev_state, key, batch_metrics = self.batch_rollout(input_tokens, seq_lens, key, prev_state)
                gather_metrics = self.gather_rollouts()

                self.prompt_id_offset = (self.prompt_id_offset + self.max_prefill_prompts) % 1048576

            metrics = (
                batch_metrics
                | gather_metrics
                | prompt_metrics
                | {"total_time": t.data["time"], "worker_id": self.global_worker_id}
            )
            logger.info(
                f"[throughput] worker={self.global_worker_id} decode_tps={metrics['decode_tps']:.0f} "
                f"decode_sps={metrics['decode_sps']:.2f} decode_time={metrics['decode_time']:.1f}s "
                f"decode_steps={metrics['decode_steps']}",
                log_for_all=True,
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
    ) -> tuple[InferenceState, Array, dict[str, float]]:
        x_batch_sharded = jax.device_put(batch_tokens, self.sharding)
        seq_lens_sharded = jax.device_put(seq_lens, self.sharding)

        prefill_state, next_key, prefill_metrics = self.prefill_step(
            x_batch_sharded, seq_lens_sharded, self.params, key
        )
        prev_state, decode_metrics = self.continuous_batch(prefill_state, prev_state)

        return prev_state, next_key, prefill_metrics | decode_metrics

    def prefill_step(
        self, input_tokens: Array, seq_lens: Array, params: PyTree, key: Array
    ) -> tuple[InferenceState, Array, dict[str, float]]:
        if self.prefill_fns.get(precompiled_length := input_tokens.shape[1]) is None:
            self.prefill_fns[precompiled_length] = jax.jit(self.prefill)

        with Tracker(timer=True) as t:
            out, next_key = self.prefill_fns[precompiled_length](
                input_tokens, seq_lens, params, key, self.prompt_id_offset
            )
            jax.tree.map(lambda x: x.block_until_ready(), (out, next_key))
        return out, next_key, {"ttft": t.data["time"]}

    def prefill(
        self, input_tokens: Array, seq_lens: Array, params: PyTree, key: Array, prompt_id_offset: int
    ) -> tuple[InferenceState, Array]:
        logger.info(f"Compiling prefill for sequence length {input_tokens.shape[1]}", log_for_all=True)

        max_prefill_prompts = self.max_prefill_prompts
        kv_cache_dtype = self.config.kv_cache_dtype
        next_key, key = jax.random.split(key)

        with jax.named_scope("prefill"):
            kv_cache = self.model.init_kv_cache(
                max_prefill_prompts,
                length=self.max_attention_length,
                dtype=kv_cache_dtype,
                sharding=self.kv_cache_sharding,
            )
            _hidden, out_cache = self.model.apply(
                params,
                x=jnp.roll(input_tokens, 1, axis=-1),
                sequence_lens=seq_lens - 1,
                kv_cache=kv_cache,
                apply_lm_head=False,
            )
            out_tokens = (
                jnp.ones((max_prefill_prompts, self.max_attention_length), dtype=jnp.int32, out_sharding=self.sharding)
                * self.pad_token
            )
            out_logprobs = jnp.zeros(
                (max_prefill_prompts, self.max_attention_length), dtype=jnp.float32, out_sharding=self.sharding
            )
            out_tokens = jax.lax.dynamic_update_slice_in_dim(out_tokens, input_tokens, 1, axis=1)
            out_logprobs = jax.lax.dynamic_update_slice_in_dim(
                out_logprobs,
                -jnp.inf * jnp.ones(input_tokens.shape, dtype=jnp.float32, out_sharding=self.sharding),
                1,
                axis=1,
            )

        state = InferenceState(
            next_token=input_tokens[:, -1:],
            seq_lens=seq_lens,
            kv_cache=out_cache,
            key=key,
            stop_mask=jnp.zeros((max_prefill_prompts, 1), dtype=bool, out_sharding=self.sharding),
            end_of_think=jnp.zeros((max_prefill_prompts, 1), dtype=bool, out_sharding=self.sharding),
            out_tokens=out_tokens,
            out_logprobs=out_logprobs,
            prompt_id=jnp.arange(input_tokens.shape[0], out_sharding=self.sharding)[:, None] + prompt_id_offset,
        )
        return state, next_key

    def continuous_batch(
        self, prefill_prompts: InferenceState, state: InferenceState | None
    ) -> tuple[InferenceState, dict[str, float]]:
        if self.decode_fn is None:
            self.decode_fn = jax.jit(self._decode_single_loop, donate_argnums=(0,))

        global_ids = jax.device_get(prefill_prompts.prompt_id).flatten().tolist()
        local_to_global = {i: gid for i, gid in enumerate(global_ids)}

        prompt_queue: list[int] = [
            i
            for i in range(self.max_prefill_prompts)
            for _ in range(self.global_rollouts[local_to_global[i]].n_rollouts)
        ]

        finished = {"tokens": [], "logprobs": [], "prompt_ids": []}
        steps = []

        with Tracker(timer=True) as t:
            if state is None:
                ids = [prompt_queue.pop() for _ in range(self.config.max_decode_batch_size)]
                state = self.create_initial_state(prefill_prompts, ids)
                for i in ids:
                    self.global_rollouts[local_to_global[i]].weight_iteration.append(self.weight_iteration)

            while len(prompt_queue) > 0:
                next_index = prompt_queue.pop()

                (state, tokens, logprobs, prompt_ids, n_steps) = self.decode_fn(
                    state, self.params, prefill_prompts, next_index
                )

                finished["tokens"].append(tokens)
                finished["logprobs"].append(logprobs)
                finished["prompt_ids"].append(prompt_ids)

                self._maybe_update_params(state)

                self.global_rollouts[local_to_global[next_index]].weight_iteration.append(self.weight_iteration)
                steps.append(n_steps)

            gathered = {k: np.concatenate([jax.device_get(x) for x in v], axis=0) for k, v in finished.items()}

        for i in range(gathered["tokens"].shape[0]):
            pid = gathered["prompt_ids"][i].item()
            self.global_rollouts[pid].rollout_tokens.append(gathered["tokens"][i])
            self.global_rollouts[pid].rollout_logprobs.append(gathered["logprobs"][i])

        queued_steps: int = sum(int(jax.device_get(s)) for s in steps)
        subbed_steps = len(steps)

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

    def warmup(self):
        with Tracker(timer=True) as t:
            key = jax.device_put(jax.random.PRNGKey(0), self.sharding)
            n = self.max_prefill_prompts
            seq_len = self.config.min_prefill_length
            while seq_len <= min(self.config.warmup_seq_len, self.max_attention_length):
                logger.info(f"Warming up prefill for sequence length {seq_len}", log_for_all=True)
                tokens = jax.device_put(np.full((n, seq_len), self.pad_token, dtype=np.int32), self.sharding)
                seq_lens = jax.device_put(np.full((n,), seq_len, dtype=np.int32), self.sharding)
                prefill_state, key, _ = self.prefill_step(tokens, seq_lens, self.params, key)
                seq_len *= 2

            logger.info("Warming up decode", log_for_all=True)
            state = self.create_initial_state(prefill_state, list(range(self.config.max_decode_batch_size)))
            state = state.replace(seq_lens=jnp.full_like(state.seq_lens, self.max_attention_length - 1))  # type: ignore
            self.decode_fn = jax.jit(self._decode_single_loop, donate_argnums=(0,))
            jax.block_until_ready(self.decode_fn(state, self.params, prefill_state, 0))

        logger.info(f"Warmup finished in {t.data['time']:.1f}s", log_for_all=True)

    def create_initial_state(self, prefill_prompts: InferenceState, initial_ids: list[int]) -> InferenceState:
        if self.initial_state_fn is None:
            self.initial_state_fn = jax.jit(self._create_initial_state)
        ids = jax.device_put(np.array(initial_ids, dtype=np.int32), self.sharding)
        return self.initial_state_fn(prefill_prompts, ids)

    def _create_initial_state(self, prefill_prompts: InferenceState, initial_ids: Array) -> InferenceState:
        def create(prompts, i):
            state = self.create_batch(prompts, i)
            return state.roll(state.seq_lens - 1)

        stacked_state = jax.vmap(create, in_axes=(None, 0))(prefill_prompts, initial_ids)
        return stacked_state.replace(
            next_token=stacked_state.next_token[:, 0],
            kv_cache=[KVCache(k=k.k[:, 0], v=k.v[:, 0], length=k.length[:, 0]) for k in stacked_state.kv_cache],
            key=stacked_state.key[0],
            seq_lens=stacked_state.seq_lens[:, 0],
            stop_mask=stacked_state.stop_mask[:, 0],
            end_of_think=stacked_state.end_of_think[:, 0],
            out_tokens=stacked_state.out_tokens[:, 0],
            out_logprobs=stacked_state.out_logprobs[:, 0],
            prompt_id=stacked_state.prompt_id[:, 0],
        )

    def _decode_single_loop(
        self, state: InferenceState, params: PyTree, prefill_prompts: InferenceState, next_index: int
    ) -> tuple[InferenceState, Array, Array, Array, Array]:
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
            next_batch = self.create_batch(prefill_prompts, next_index)
            state = self.sub_batch(state, next_batch)

        return state, tokens, logprobs, prompt_ids, jnp.max(after_length - before_length)

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

        with jax.named_scope("masking"):
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

        rows = jnp.arange(state.out_tokens.shape[0])
        write_index = out_cache[0].length
        out_tokens = state.out_tokens.at[rows, write_index].set(next_token[:, 0])
        out_logprobs = state.out_logprobs.at[rows, write_index].set(next_log_prob[:, 0])

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

    def create_batch(self, prefill_prompts: InferenceState, index: int | Array) -> InferenceState:
        def ds(x):
            return jax.lax.dynamic_slice_in_dim(x, index, 1, axis=0)

        return InferenceState(
            next_token=ds(prefill_prompts.next_token),
            kv_cache=[
                KVCache(k=ds(cache.k), v=ds(cache.v), length=ds(cache.length)) for cache in prefill_prompts.kv_cache
            ],
            key=prefill_prompts.key,
            seq_lens=ds(prefill_prompts.seq_lens),
            stop_mask=ds(prefill_prompts.stop_mask),
            end_of_think=ds(prefill_prompts.end_of_think),
            out_tokens=ds(prefill_prompts.out_tokens),
            out_logprobs=ds(prefill_prompts.out_logprobs),
            prompt_id=ds(prefill_prompts.prompt_id),
        )

    def sub_batch(self, state: InferenceState, new_batch: InferenceState) -> InferenceState:
        index = jnp.argmax(state.stop_mask[:, 0], keepdims=True)
        new_batch = new_batch.roll(new_batch.seq_lens - 1)
        return state.sub(new_batch, index)

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

    def _maybe_update_params(self, state: InferenceState | None = None):
        if state is not None:
            jax.block_until_ready(state)

        if not self.async_state.ready_to_sync.is_set():
            return

        with self.async_state.worker_signal:
            self.async_state.ready_workers += 1
            self.async_state.worker_signal.notify_all()

        while self.async_state.ready_to_sync.is_set():
            time.sleep(0.1)

        with self.async_state.read_write_lock:
            self.params = jax.tree.map(
                lambda x: jax.make_array_from_single_device_arrays(
                    x.shape,
                    self.sharding,
                    arrays=[s.data for s in x.addressable_shards if s.device in self.devices],
                    dtype=x.dtype,
                ),
                self.async_state.MRUparams,
            )
            self.weight_iteration = self.async_state.weight_iteration

        logger.info(f"Params updated on inference worker {self.global_worker_id}", log_for_all=True)

    def block_until_params_update(self):
        logger.info("Waiting for initial parameters from training workers...", log_for_all=True)
        while not self.async_state.ready_to_sync.is_set():
            time.sleep(0.2)
        self._maybe_update_params()

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
        padding_length = max(self.compute_max_padding_length(seq_lens), self.config.min_prefill_length)
        inputs = [(padding_length - len(x)) * [self.pad_token] + x for x in inputs]
        tokens = np.array(inputs, dtype=np.int32)

        return tokens, seq_lens

    def compute_max_padding_length(self, seq_lens: np.ndarray) -> int:
        return self.compute_max_power_of_two(max(seq_lens).item(), self.max_attention_length)

    def compute_max_power_of_two(self, n: int, upper_bound: int) -> int:
        return min(1 << (n.bit_length()), upper_bound)

    @property
    def max_prefill_prompts(self) -> int:
        return self.config.prefill_multiplier * self.config.max_decode_batch_size

    @property
    def max_attention_length(self) -> int:
        return min(self.config.max_seq_len, self.model.sequence_len)


class AsyncInferenceWorker(Worker):
    def __init__(self, trainer_config: TrainerConfig, async_options: AsyncOptions):
        self.trainer_config = trainer_config
        self.inference_config = trainer_config.loss_config.inference_config
        self.eval_config = trainer_config.eval_config
        self.async_options = async_options

        self.model = Model(trainer_config.model_config)
        self.validate_config()

        abstract_output = self.model.init_state(rng=jax.random.PRNGKey(0), tx=None, sharding=None, abstract=True)
        dummy_params = jax.tree.map(
            lambda x: jnp.zeros(x.shape, dtype=self.inference_config.params_dtype), abstract_output
        )

        self.mesh = jax.sharding.Mesh(
            np.array(jax.local_devices()), axis_names=(GLOBAL_AXIS,), axis_types=(jax.sharding.AxisType.Explicit,)
        )
        self.sharding = jax.NamedSharding(self.mesh, P())

        self.params = jax.device_put(dummy_params, self.sharding)

        self.worker_rank = stax.get_rank() - self.async_options.train_workers
        self.async_state = AsyncState(MRUparams=self.params)
        self.transfer_server = RDMATransferServer(train_workers=self.async_options.train_workers)
        self.monitor_thread = threading.Thread(
            target=self.monitor_weight_sync, args=(self.async_state, self.async_options), daemon=True
        )
        self.monitor_thread.start()

        device_groups = np.array(jax.local_devices()).reshape(-1, self.inference_config.tp)
        self.n_replicas = device_groups.shape[0]

        workers = [
            SingleInferenceReplica(
                self.inference_config,
                self.model,
                self.async_state,
                async_options,
                i,
                self.worker_rank,
                device_groups[i],
                eval_group_size=self.eval_config.group_size,
            )
            for i in range(self.n_replicas)
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

            with async_state.worker_signal:
                async_state.ready_workers = 0
                async_state.ready_to_sync.set()
                async_state.worker_signal.wait_for(lambda: async_state.ready_workers >= self.n_replicas, timeout=60)
                if async_state.ready_workers < self.n_replicas:
                    logger.warning(f"Got {async_state.ready_workers} signals, expected {self.n_replicas}")
            new_params = self.transfer_server.transfer(async_state.MRUparams)

            with async_state.read_write_lock:
                async_state.MRUparams = new_params
                async_state.weight_iteration += 1

            async_state.ready_to_sync.clear()

    def validate_config(self):
        assert jax.local_device_count() % self.inference_config.tp == 0, (
            f"tp {self.inference_config.tp} must divide local device count {jax.local_device_count()}"
        )
        assert self.inference_config.max_seq_len <= self.model.sequence_len, (
            f"expected inference max seq len {self.inference_config.max_seq_len} to be less than model sequence length {self.model.sequence_len}"
        )
        assert self.inference_config.max_seq_len & (self.inference_config.max_seq_len - 1) == 0, (
            f"max_seq_len must be a power of 2, got {self.inference_config.max_seq_len}"
        )
        assert self.inference_config.min_prefill_length & (self.inference_config.min_prefill_length - 1) == 0, (
            f"min_prefill_length must be a power of 2, got {self.inference_config.min_prefill_length}"
        )
        assert self.inference_config.min_prefill_length >= 128, (
            "Flash-attention prefill requires min prefill length of 128"
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

        assert self.inference_config.prefill_multiplier >= 1, (
            f"prefill_multiplier must be >= 1, got {self.inference_config.prefill_multiplier}"
        )

        if self.inference_config.warmup_seq_len is not None:
            warmup_seq_len = self.inference_config.warmup_seq_len
            assert warmup_seq_len & (warmup_seq_len - 1) == 0, (
                f"warmup_seq_len must be a power of 2, got {warmup_seq_len}"
            )

        if self.inference_config.top_k is not None:
            assert self.inference_config.top_k > 0, f"top_k must be positive, got {self.inference_config.top_k}"

        if self.inference_config.top_p is not None:
            assert 0.0 < self.inference_config.top_p <= 1.0, (
                f"top_p must be in the range (0, 1], got {self.inference_config.top_p}"
            )

        if self.inference_config.tp > 1:
            raise NotImplementedError("tp >1 is not supported right now")
