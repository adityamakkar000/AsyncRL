import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree
from stax.logger import staxLogger as logger
from transformers import AutoTokenizer

from src.model import KVCache, Model, ModelConfig, QwenConfig

from .config import InferenceConfig, InferenceResults, InferenceRollout, InferenceState

"""
#TODO: Inference 
- integrate into trainer 
- fix key splitting across devices
"""


class InferenceEngine:
    def __init__(self, model: Model, params: PyTree, config: InferenceConfig):
        self.model = model
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.batch_size = config.batch_size * jax.local_device_count()
        self.group_size = config.group_size
        self.num_prompts = self.batch_size // self.group_size
        self.inital_sequence_len = 64

        self.mesh = jax.make_mesh((jax.local_device_count(),), ("data",), devices=jax.local_devices())

        assert self.max_seq_len <= self.model.sequence_len, (
            f"expcted inference max seq len {self.max_seq_len} to be less than model sequence length {self.model.sequence_len}"
        )
        assert self.max_seq_len & (self.max_seq_len - 1) == 0, (
            f"max_seq_len must be a power of 2, got {self.max_seq_len}"
        )
        assert self.config.group_size % self.batch_size == 0, (
            "Batch size must be divisible by group size for static batching"
        )

        if self.config.top_k is not None:
            assert self.config.top_k > 0, f"top_k must be positive, got {self.config.top_k}"

        if self.config.top_p is not None:
            assert 0.0 < self.config.top_p <= 1.0, f"top_p must be in the range (0, 1], got {self.config.top_p}"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.hf_model_name)

        self.precompile_dict = {
            "prefill": {},
            "decode": {},
        }

        if self.config.precompile:
            self.precompile_prefill(params)
            self.precompile_decode(params)

    def replicate_across_axis(self, value):
        return jax.device_put(value, jax.sharding.NamedSharding(self.mesh, P()))

    def split_across_axis(self, value):
        return jax.device_put(value, jax.sharding.NamedSharding(self.mesh, P(self.mesh.axis_names[0])))

    def precompile_prefill(self, params: PyTree) -> None:
        """Precompile prefill function for different sequence lengths up to max_seq_len.
        The structure self.precompiled_dict = dict[seq_len --> compiled_fn]."""

        curr_seq_len = self.inital_sequence_len
        key = jax.random.PRNGKey(0)
        seq_lens = jnp.array([1] * self.batch_size)
        kv_cache = self.model.init_kv_cache(self.batch_size, dtype=self.config.kv_cache_dtype, mesh=self.mesh)

        params, key = self.replicate_across_axis(params), self.replicate_across_axis(key)
        seq_lens = self.split_across_axis(seq_lens)

        while curr_seq_len <= self.max_seq_len:
            x_init = self.split_across_axis(jnp.ones((self.batch_size, curr_seq_len), dtype=jnp.int32))
            state = InferenceState(
                next_token=self.split_across_axis(jnp.ones((self.batch_size, 1), dtype=jnp.int32)),
                next_probs=self.split_across_axis(jnp.ones((self.batch_size, 1))),
                kv_cache=kv_cache,
                key=key,
                seq_lens=seq_lens,
                params=params,
                stop_mask=self.split_across_axis(jnp.zeros((self.batch_size, 1), dtype=bool)),
            )

            in_shardings = jax.tree.map(lambda x: x.sharding, (x_init, state))
            out_shardings = jax.tree.map(lambda x: x.sharding, state)

            self.precompile_dict["prefill"][curr_seq_len] = jax.jit(
                self.prefill, in_shardings=in_shardings, out_shardings=out_shardings
            )  # float32[B, T] float32[B, T]['data']
            _output = self.precompile_dict["prefill"][curr_seq_len](x_init, state)
            curr_seq_len *= 2
        logger.info("Finished prefill precompile")

    def precompile_decode(self, params: PyTree) -> None:
        curr_size = self.inital_sequence_len

        state = InferenceState(
            next_token=self.split_across_axis(jnp.ones((self.batch_size, 1), dtype=jnp.int32)),
            next_probs=self.split_across_axis(jnp.ones((self.batch_size, 1))),
            kv_cache=self.model.init_kv_cache(self.batch_size, dtype=self.config.kv_cache_dtype, mesh=self.mesh),
            key=self.replicate_across_axis(jax.random.PRNGKey(0)),
            seq_lens=self.split_across_axis(jnp.array([1] * self.batch_size)),
            params=self.replicate_across_axis(params),
            stop_mask=self.split_across_axis(jnp.zeros((self.batch_size, 1), dtype=bool)),
        )

        in_shardings = out_shardings = jax.tree.map(lambda x: x.sharding, state)
        while curr_size <= self.max_seq_len:
            self.precompile_dict["decode"][curr_size] = jax.jit(
                lambda state: self.decode(state, curr_size),
                donate_argnums=(0,),
                in_shardings=(in_shardings,),
                out_shardings=out_shardings,
            )
            _ = self.precompile_dict["decode"][curr_size](state)
            curr_size *= 2
        logger.info("Finished decode precompile")

    def compute_max_power_of_two(self, n: int, upper_bound: int) -> int:
        return min(1 << (n.bit_length()), upper_bound)

    def compute_max_padding_length(self, seq_lens: Array) -> int:
        return self.compute_max_power_of_two(jnp.max(seq_lens).item(), self.max_seq_len)

    def compute_attention_length(self, cache_length: Array) -> int:
        return self.compute_max_power_of_two(cache_length.item(), self.model.config.qwen_config.sequence_len)

    def tokenize(self, texts: list[str]) -> tuple[Array, Array]:
        inputs: list[list[int]] = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=True
            )
            for text in texts
        ]
        seq_lens = jnp.array([len(x) for x in inputs], dtype=jnp.int32)
        padding_length = max(self.compute_max_padding_length(seq_lens), self.inital_sequence_len)
        inputs = [(padding_length - len(x)) * [self.tokenizer.pad_token_id] + x for x in inputs]
        tokens = jnp.array(inputs)

        return tokens, jnp.array(seq_lens, dtype=jnp.int32)

    def cleanup_rollouts(self, rollouts: list[InferenceRollout]) -> list[InferenceRollout]:
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

    def detokenizer(self, tokens: list[InferenceRollout] | InferenceRollout) -> list[list[str]] | list[str]:
        if isinstance(tokens, InferenceRollout):
            tokens = [tokens]

        detokenized = []
        for rollout in tokens:
            rollout_strs = self.tokenizer.batch_decode(rollout.rollouts, skip_special_tokens=False)
            detokenized.append(rollout_strs)

        return detokenized if len(detokenized) > 1 else detokenized[0]

    def sample_logits(self, logits: Array, key: Array) -> tuple[Array, Array]:
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

    def prefill(self, input_tokens: Array, state: InferenceState) -> InferenceState:
        logger.info(f"Compiling prefill for sequence length {input_tokens.shape[1]}")
        logits, out_cache = self.model.apply(
            state.params, x=input_tokens[:, :-1], sequence_lens=state.seq_lens - 1, kv_cache=state.kv_cache
        )
        return state.replace(  # type: ignore : state is of flax dataclass and replace is an function
            next_token=input_tokens[:, -1:],
            next_probs=jnp.ones((self.batch_size, 1)),
            kv_cache=out_cache,
        )

    def decode(self, state: InferenceState, attention_length: int) -> InferenceState:
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
        stop_mask = (
            state.stop_mask
            | (next_token == self.tokenizer.eos_token_id)
            | (state.seq_lens[:, None] + 1 > self.max_seq_len)
        )
        return InferenceState(
            next_token=jnp.where(stop_mask, self.tokenizer.eos_token_id, next_token),
            next_probs=jnp.where(stop_mask, 0, next_log_prob),
            kv_cache=out_cache,
            key=key,
            seq_lens=state.seq_lens + jnp.where(stop_mask, 0, 1)[:, 0],
            stop_mask=stop_mask,
            params=state.params,
        )

    def prefill_step(self, input_tokens: Array, inference_state: InferenceState) -> InferenceState:
        precompiled_length = input_tokens.shape[1]
        if self.precompile_dict["prefill"].get(precompiled_length) is None:
            in_shardings = jax.tree.map(lambda x: x.sharding, (input_tokens, inference_state))
            out_shardings = jax.tree.map(lambda x: x.sharding, inference_state)
            self.precompile_dict["prefill"][precompiled_length] = jax.jit(
                self.prefill, in_shardings=in_shardings, out_shardings=out_shardings
            )

        return self.precompile_dict["prefill"][precompiled_length](input_tokens, inference_state)

    def decode_step(
        self, inference_state: InferenceState, out_tokens: Array, out_logprobs: Array
    ) -> tuple[InferenceState, Array, Array]:
        attention_length = self.compute_attention_length(inference_state.kv_cache[0].length)  # type: ignore
        if self.precompile_dict["decode"].get(attention_length) is None:
            in_shardings = out_shardings = jax.tree.map(lambda x: x.sharding, inference_state)
            self.precompile_dict["decode"][attention_length] = jax.jit(
                lambda state: self.decode(state, attention_length),
                donate_argnums=(0,),
                in_shardings=(in_shardings,),
                out_shardings=out_shardings,
            )
        new_state: InferenceState = self.precompile_dict["decode"][attention_length](inference_state)
        out_tokens = jnp.concat((out_tokens, new_state.next_token[...]), axis=-1)
        out_logprobs = jnp.concat((out_logprobs, new_state.next_probs[...]), axis=-1)
        return new_state, out_tokens, out_logprobs

    def get_decode_batch(self, batch: int, step: int, intial_state: InferenceState) -> InferenceState:
        def repeat_to_batch(x: Array) -> Array:
            return jnp.copy(jnp.repeat(x, self.batch_size, axis=0))

        kv_cache = [
            KVCache(
                k=repeat_to_batch(layer_cache.k[batch : batch + 1, ...]),
                v=repeat_to_batch(layer_cache.v[batch : batch + 1, ...]),
                length=jnp.copy(layer_cache.length),  # type: ignore
            )
            for layer_cache in intial_state.kv_cache
        ]

        return InferenceState(
            next_token=repeat_to_batch(intial_state.next_token[batch : batch + 1, :]),
            next_probs=repeat_to_batch(intial_state.next_probs[batch : batch + 1, :]),
            kv_cache=kv_cache,
            key=jax.random.fold_in(intial_state.key, batch * step + step),
            seq_lens=repeat_to_batch(intial_state.seq_lens[batch : batch + 1]),
            params=jax.tree.map(lambda x: jnp.copy(x), intial_state.params),
            stop_mask=repeat_to_batch(intial_state.stop_mask[batch : batch + 1, :]),
        )

    def single_batch_decode(self, inference_state: InferenceState, prompt_tokens: Array) -> tuple[Array, Array]:
        out_tokens = prompt_tokens
        out_logprobs = jnp.zeros_like(prompt_tokens, dtype=jnp.float32)
        while not jnp.all(inference_state.stop_mask):
            inference_state, out_tokens, out_logprobs = self.decode_step(inference_state, out_tokens, out_logprobs)
        return jax.tree.map(lambda x: list(jax.device_get(x)), (out_tokens, out_logprobs))

    def batch_decode(self, x: Array, intial_state: InferenceState) -> tuple[list[InferenceRollout], PyTree]:
        assert (B := x.shape[0]) == self.batch_size, f"Expected batch size {self.batch_size}, got {B}"

        output_rollouts = [InferenceRollout(rollouts=[], logprobs=[]) for _ in range(self.batch_size)]

        prefill_state = self.prefill_step(x, intial_state)

        for batch in range(self.batch_size):
            for step in range(self.config.group_size // self.batch_size):
                inference_state = self.get_decode_batch(batch, step, prefill_state)
                prompt_tokens = jnp.repeat(x[batch : batch + 1, :], self.batch_size, axis=0)
                output_tokens, output_logprobs = self.single_batch_decode(inference_state, prompt_tokens)

                output_rollouts[batch].rollouts.extend(output_tokens)
                output_rollouts[batch].logprobs.extend(output_logprobs)

        metrics = {}
        return output_rollouts, metrics

    def multi_batch_decode(
        self, batch_tokens: Array, seq_lens: Array, key: Array, params: PyTree
    ) -> tuple[list[InferenceRollout], PyTree]:
        B, _ = batch_tokens.shape
        batches = B // self.batch_size
        output = []
        metrics = []

        key, params = self.replicate_across_axis(key), self.replicate_across_axis(params)
        batch_tokens, seq_lens = self.split_across_axis(batch_tokens), self.split_across_axis(seq_lens)
        next_token = self.split_across_axis(jnp.zeros((self.batch_size, 1), dtype=jnp.int32))
        next_probs = self.split_across_axis(jnp.zeros((self.batch_size, 1)))
        stop_mask = self.split_across_axis(jnp.zeros((self.batch_size, 1), dtype=bool))

        for i in range(batches):
            tokens = batch_tokens[i * self.batch_size : (i + 1) * self.batch_size]
            seq_lens_batch = seq_lens[i * self.batch_size : (i + 1) * self.batch_size]
            intial_state = InferenceState(
                next_token=next_token,
                next_probs=next_probs,
                kv_cache=self.model.init_kv_cache(self.batch_size, dtype=self.config.kv_cache_dtype, mesh=self.mesh),
                key=jax.random.fold_in(key, i),
                seq_lens=seq_lens_batch,
                params=params,
                stop_mask=stop_mask,
            )

            batch_output, batch_metrics = self.batch_decode(tokens, intial_state)
            output.extend(batch_output)
            metrics.append(batch_metrics)

        metrics = jax.tree.map(lambda *x: jnp.stack(x).mean(axis=0), *metrics)
        return self.cleanup_rollouts(output), metrics

    def pad_to_batch_size(self, batch_tokens: Array, seq_lens: Array) -> tuple[Array, Array]:
        B, T = batch_tokens.shape
        if B % self.batch_size == 0:
            return batch_tokens, seq_lens

        pad_size = self.batch_size - (B % self.batch_size)
        padded_inputs = jnp.concatenate(
            (batch_tokens, jnp.full((pad_size, T), self.tokenizer.pad_token_id, dtype=jnp.int32)), axis=0
        )
        padded_seq_lens = jnp.concatenate((seq_lens, jnp.array([T] * pad_size)), axis=0)
        return padded_inputs, padded_seq_lens

    def rollout(
        self, batch_tokens: Array, seq_lens: Array, key: Array, params: PyTree
    ) -> tuple[list[InferenceRollout], PyTree]:
        padded_batch_tokens, padded_seq_lens = self.pad_to_batch_size(batch_tokens, seq_lens)
        return self.multi_batch_decode(padded_batch_tokens, padded_seq_lens, key, params)

    def __call__(self, prompts: list[str], key: Array, params: PyTree, detokenize: bool = False) -> InferenceResults:
        inp_tokens, seq_lens = self.tokenize(prompts)
        output_rollouts, metrics = self.rollout(inp_tokens, seq_lens, key, params)
        output_strs = self.detokenizer(output_rollouts) if detokenize else None
        return InferenceResults(rollouts=output_rollouts, output_strs=output_strs, metrics=metrics)


if __name__ == "__main__":
    model_config = ModelConfig(
        "Qwen/Qwen3-0.6B",
        qwen_config=QwenConfig(
            vocab_size=151936,
            d_ff=3072,
            sequence_len=16_384,
            model_dim=1024,
            n_heads=16,
            n_groups=8,
            head_dim=128,
            n_layers=28,
            rope_base=1000000,
            activation_dtype="float32",
        ),
    )
    model = Model(model_config)

    params = model.init_state(jax.random.PRNGKey(0), None, abstract=False)
    config = InferenceConfig(
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_seq_len=128,
        batch_size=2,
        group_size=8,
        kv_cache_dtype="bfloat16",
        precompile=False,
    )

    engine = InferenceEngine(model, params, config)

    key = jax.random.PRNGKey(2303)
    params = model.init_state(jax.random.PRNGKey(0), None, abstract=False)

    tokenizer_inp = ["What is 2 + 2", "Explain the theory of relativity", "what is 3 + 3", "what is 4 + 4"]

    # inp_tokens, sequence_lens = engine.tokenize(tokenizer_inp)

    # key = jax.random.PRNGKey(2303)
    # output_rollouts, metrics = engine.rollout(inp_tokens, sequence_lens, key, params)

    output = engine(tokenizer_inp, key, params, detokenize=True)
    breakpoint()
