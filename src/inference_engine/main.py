import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from stax.logger import staxLogger as logger
from transformers import AutoTokenizer

from src.model import KVCache, Model, ModelConfig, QwenConfig

from .config import InferenceConfig, InferenceRollout, InferenceState

"""
#TODO: Inference 
- multihost 
- fix up static batching
- integrate dataclass for return outputs from prefill  (done)
- precompile along batch and T for prefill
- precompile decode along batch 
- max sequence length + stop token breaking
- donate kv cache memory optimization 
- setup inference loop
- integrate tokenizer into single call function (done)
- return back prob tokens
- precompile attention length so no need to do full 16k for every turn
"""


"""

1. return back logprobs
2. self.batch_decode always same batch size + fix multi batch decode 
3. rollout G groups in the resonse and convert otuput to list[InferenceRollout]
4. stopping mechanism based on eos token or max length

"""


class InferenceEngine:
    def __init__(self, model: Model, params: PyTree, config: InferenceConfig):
        self.model = model
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.batch_size = config.batch_size
        self.group_size = config.group_size
        self.num_prompts = self.batch_size // self.group_size
        self.inital_sequence_len = 64

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

    def precompile_prefill(self, params: PyTree) -> None:
        """Precompile prefill function for different sequence lengths up to max_seq_len.
        The structure self.precompiled_dict = dict[seq_len --> compiled_fn]."""

        curr_seq_len = self.inital_sequence_len
        key = jax.random.PRNGKey(0)
        seq_lens = jnp.array([1] * self.num_prompts)
        kv_cache = self.model.init_kv_cache(self.batch_size, dtype=self.config.kv_cache_dtype)

        while curr_seq_len <= self.max_seq_len:
            x_init = jnp.ones((self.num_prompts, curr_seq_len), dtype=jnp.int32)
            state = InferenceState(
                next_token=jnp.ones((self.num_prompts, 1), dtype=jnp.int32),
                next_probs=jnp.ones((self.num_prompts, 1)),
                kv_cache=kv_cache,
                key=key,
                seq_lens=seq_lens,
                params=params,
                stop_mask=jnp.zeros((self.num_prompts, 1), dtype=bool),
            )

            self.precompile_dict["prefill"][curr_seq_len] = jax.jit(self.prefill)
            _output = self.precompile_dict["prefill"][curr_seq_len](x_init, state)
            curr_seq_len *= 2
        logger.info("Finished prefill precompile")

    def precompile_decode(self, params: PyTree) -> None:
        curr_size = self.inital_sequence_len

        state = InferenceState(
            next_token=jnp.ones((self.batch_size, 1), dtype=jnp.int32),
            next_probs=jnp.ones((self.batch_size, 1)),
            kv_cache=self.model.init_kv_cache(self.batch_size, dtype=self.config.kv_cache_dtype),
            key=jax.random.PRNGKey(0),
            seq_lens=jnp.array([1] * self.batch_size),
            params=params,
            stop_mask=jnp.zeros((self.batch_size, 1), dtype=bool),
        )
        while curr_size <= self.max_seq_len:
            self.precompile_dict["decode"][curr_size] = jax.jit(lambda state: self.decode(state, curr_size))
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

    def detokenizer(self, batch_tokens: list[list[int]]) -> list[str]:
        def remove_padding(tokens: list[int]) -> list[int]:
            start_idx = next((i for i, token in enumerate(tokens) if token != self.tokenizer.pad_token_id), len(tokens))
            end_idx = (
                len(tokens)
                - next(
                    (i for i, token in enumerate(reversed(tokens)) if token != self.tokenizer.eos_token_id), len(tokens)
                )
                + 1
            )
            if end_idx <= start_idx:
                logger.warning("All tokens are padding, returning empty string")
                return []
            return tokens[start_idx:end_idx]

        return self.tokenizer.batch_decode(
            [remove_padding(tokens) for tokens in batch_tokens], skip_special_tokens=False
        )

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
        return state.replace(  # type: ignore : state is of flax dataclass and replace is an function
            next_token=jnp.where(stop_mask, self.tokenizer.eos_token_id, next_token),
            next_probs=jnp.where(stop_mask, 0, next_log_prob),
            kv_cache=out_cache,
            key=key,
            seq_lens=state.seq_lens + jnp.where(stop_mask, 0, 1)[:, 0],
            stop_mask=stop_mask,
        )

    def prefill_step(self, input_tokens: Array, inference_state: InferenceState) -> InferenceState:
        precompiled_length = input_tokens.shape[1]
        if self.precompile_dict["prefill"].get(precompiled_length) is None:
            self.precompile_dict["prefill"][precompiled_length] = jax.jit(self.prefill)

        return self.precompile_dict["prefill"][precompiled_length](input_tokens, inference_state)

    def decode_step(
        self, inference_state: InferenceState, attention_length: int, out_tokens: Array
    ) -> tuple[Array, InferenceState]:
        if self.precompile_dict["decode"].get(attention_length) is None:
            self.precompile_dict["decode"][attention_length] = jax.jit(
                lambda state: self.decode(state, attention_length)
            )
        inference_state = self.precompile_dict["decode"][attention_length](inference_state)
        out_tokens = jnp.concat((out_tokens, inference_state.next_token), axis=-1)
        return out_tokens, inference_state

    def get_decode_batch(self, batch: int, step: int, intial_state: InferenceState) -> InferenceState:
        def repeat_to_batch(x: Array) -> Array:
            return jnp.repeat(x, self.batch_size, axis=0)

        kv_cache = [
            KVCache(
                k=repeat_to_batch(layer_cache.k[batch : batch + 1, ...]),
                v=repeat_to_batch(layer_cache.v[batch : batch + 1, ...]),
                length=layer_cache.length,
            )
            for layer_cache in intial_state.kv_cache
        ]

        return InferenceState(
            next_token=repeat_to_batch(intial_state.next_token[batch : batch + 1, :]),
            kv_cache=kv_cache,
            key=jax.random.fold_in(intial_state.key, batch * step + step),
            seq_lens=repeat_to_batch(intial_state.seq_lens[batch : batch + 1]),
            params=intial_state.params,
            stop_mask=repeat_to_batch(intial_state.stop_mask[batch : batch + 1, :]),
        )

    def single_batch_decode(self, inference_state: InferenceState, prompt_tokens: Array) -> list[int]:
        out_tokens = prompt_tokens
        while not jnp.all(inference_state.stop_mask):
            attention_len = self.compute_attention_length(inference_state.kv_cache[0].length)  # type: ignore
            out_tokens, inference_state = self.decode_step(inference_state, attention_len, out_tokens)
        return jax.device_get(out_tokens).tolist()

    def batch_decode(self, x: Array, seq_lens: Array, key: Array, params: PyTree) -> tuple[list[list[int]], PyTree]:
        assert (B := x.shape[0]) == self.batch_size, f"Expected batch size {self.batch_size}, got {B}"
        intial_state = InferenceState(
            next_token=jnp.empty((self.batch_size, 1), dtype=jnp.int32),
            next_probs=jnp.empty((self.batch_size, 1)),
            kv_cache=self.model.init_kv_cache(self.batch_size, dtype=self.config.kv_cache_dtype),
            key=key,
            seq_lens=seq_lens,
            params=params,
            stop_mask=jnp.zeros((self.batch_size, 1), dtype=bool),
        )

        output_list_tokens: list[list[list[int]]] = [[] for _ in range(self.batch_size)]

        prefill_state = self.prefill_step(x, intial_state)

        for batch in range(self.batch_size):
            for step in range(self.config.group_size // self.batch_size):
                inference_state = self.get_decode_batch(batch, step, prefill_state)
                prompt_tokens = jnp.repeat(x[batch : batch + 1, :], self.batch_size, axis=0)
                output_tokens = self.single_batch_decode(inference_state, prompt_tokens)
                output_list_tokens[batch].extend(output_tokens)

        metrics = {}
        return output_list_tokens, metrics

    def multi_batch_decode(
        self, batch_tokens: Array, seq_lens: Array, key: Array, params: PyTree
    ) -> tuple[list[InferenceRollout], PyTree]:
        B, _ = batch_tokens.shape
        batches = B // self.batch_size
        output = []
        metrics = []

        for i in range(batches):
            tokens = batch_tokens[i * self.batch_size : (i + 1) * self.batch_size]
            seq_lens_batch = seq_lens[i * self.batch_size : (i + 1) * self.batch_size]
            batch_output, batch_metrics = self.batch_decode(tokens, seq_lens_batch, key, params)
            output.append(batch_output)
            metrics.append(batch_metrics)

        metrics = jax.tree.map(lambda *x: jnp.stack(x).mean(axis=0), *metrics)
        return output, metrics

    def pad_to_batch_size(self, batch_tokens: Array, seq_lens: Array) -> tuple[Array, Array]:
        breakpoint()
        B, T = batch_tokens.shape
        if B % self.batch_size == 0:
            return batch_tokens, seq_lens

        pad_size = batch_tokens % self.batch_size
        padded_inputs = jnp.concatenate(
            (batch_tokens, jnp.full((pad_size, T), self.tokenizer.pad_token_id, dtype=jnp.int32)), axis=0
        )
        padded_seq_lens = jnp.concatenate((seq_lens, jnp.array([T] * pad_size)), axis=0)
        breakpoint()
        return padded_inputs, padded_seq_lens

    def rollout(
        self, batch_tokens: Array, seq_lens: Array, key: Array, params: PyTree
    ) -> tuple[list[InferenceRollout], PyTree]:
        padded_batch_tokens, padded_seq_lens = self.pad_to_batch_size(batch_tokens, seq_lens)
        return self.multi_batch_decode(padded_batch_tokens, padded_seq_lens, key, params)
        # return self.batch_decode(batch_tokens, seq_lens, key, params)

    def __call__(self, texts: list[str], key: Array, params: PyTree) -> tuple[list[str], Array, PyTree]:
        inp_tokens, seq_lens = self.tokenize(texts)
        output_tokens, output_logprobs, metrics = self.rollout(inp_tokens, seq_lens, key, params)
        output = self.detokenizer(output_tokens)
        return output, output_logprobs, metrics


def get_memory_usage():
    stats = jax.local_devices()[0].memory_stats()
    return stats["bytes_in_use"] / (1024**3)


if __name__ == "__main__":
    print(f"Memory inital: {get_memory_usage()}")  # 6.48e-05 GB

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
        max_seq_len=1024,
        batch_size=2,
        group_size=2,
        kv_cache_dtype="bfloat16",
        precompile=False,
    )

    engine = InferenceEngine(model, params, config)

    key = jax.random.PRNGKey(2303)
    params = model.init_state(jax.random.PRNGKey(0), None, abstract=False)

    tokenizer_inp = ["What is 2 + 2", "Explain the theory of relativity"]

    inp_tokens, sequence_lens = engine.tokenize(tokenizer_inp)

    key = jax.random.PRNGKey(2303)
    output_tokens, output_logprobs, metrics = engine.rollout(inp_tokens, sequence_lens, key, params)

    output = list(map(engine.detokenizer, output_tokens))
    print(metrics)
    breakpoint()
