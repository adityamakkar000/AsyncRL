import math
import time

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from stax.logger import staxLogger as logger
from transformers import AutoTokenizer

from src.model import Model, ModelConfig, QwenConfig

from .config import InferenceConfig, InferenceState

"""
#TODO: Inference 
- multihost 
- fix up static batching
- integrate dataclass for return outputs from prefill  (done)
- precompile along batch and T for prefill
- precompile decode along batch 
- max sequence length + stop token breaking
- roll kv cache 
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.hf_model_name)

        self.precompile_dict = {
            "prefill": {},
            "decode": {},
        }

        self.precompile_prefill(params)
        self.precompile_decode(params)

    def precompile_prefill(self, params: PyTree) -> None:
        """Precompile prefill function for different sequence lengths up to max_seq_len.
        The structure self.precompiled_dict = dict[seq_len --> compiled_fn]."""

        curr_seq_len = self.inital_sequence_len
        key = jax.random.PRNGKey(0)
        seq_lens = jnp.array([1] * self.num_prompts)
        kv_cache = self.model.init_kv_cache(jnp.ones((self.batch_size, 1)), dtype=self.config.kv_cache_dtype)

        while curr_seq_len <= self.max_seq_len:
            x_init = jnp.ones((self.num_prompts, curr_seq_len), dtype=jnp.int32)
            state = InferenceState(
                next_token=jnp.ones((self.num_prompts, 1), dtype=jnp.int32),
                kv_cache=kv_cache,
                key=key,
                seq_lens=seq_lens,
                params=params,
            )

            self.precompile_dict["prefill"][curr_seq_len] = jax.jit(self.prefill)
            _output = self.precompile_dict["prefill"][curr_seq_len](x_init, state)
            curr_seq_len *= 2
        logger.info("Finished prefill precompile")

    def precompile_decode(self, params: PyTree) -> None:
        curr_size = self.inital_sequence_len

        state = InferenceState(
            next_token=jnp.ones((self.batch_size, 1), dtype=jnp.int32),
            kv_cache=self.model.init_kv_cache(jnp.ones((self.batch_size, curr_size)), dtype=self.config.kv_cache_dtype),
            key=jax.random.PRNGKey(0),
            seq_lens=jnp.array([1] * self.batch_size),
            params=params,
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
        padding_length = self.compute_max_padding_length(seq_lens)
        inputs = [(padding_length - len(x)) * [self.tokenizer.pad_token_id] + x for x in inputs]
        tokens = jnp.array(inputs)

        return tokens, jnp.array(seq_lens, dtype=jnp.int32)

    def detokenizer(self, tokens: Array) -> list[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=False)

    def update_seq_lens(self, t: int, seq_lens: Array):
        return t + seq_lens

    def sample_logits(self, logits: Array, key: Array) -> Array:
        B, T, V = logits.shape
        logits = logits[:, -1, :] / self.config.temperature

        if self.config.top_k > 0:
            top_k = min(self.config.top_k, V)
            logits, base_indices = jax.lax.top_k(logits, top_k)
        else:
            base_indices = jnp.tile(jnp.arange(V), (B, 1))

        probs = jax.nn.softmax(logits, axis=-1)
        if self.config.top_p < 1.0:
            sort_idx = jnp.argsort(-probs, axis=-1)
            sorted_probs = jnp.take_along_axis(probs, sort_idx, axis=-1)
            sorted_logits = jnp.take_along_axis(logits, sort_idx, axis=-1)

            mask = jnp.cumsum(sorted_probs, axis=-1) <= self.config.top_p
            mask = mask.at[:, 0].set(True)

            filtered_logits = jnp.where(mask, sorted_logits, -jnp.inf)

            logits = jnp.take_along_axis(filtered_logits, jnp.argsort(sort_idx, axis=-1), axis=-1)
            probs = jax.nn.softmax(logits, axis=-1)

        next_idx = jax.random.categorical(key, logits, axis=-1)[:, None]
        next_tokens = jnp.take_along_axis(base_indices, next_idx, axis=-1)
        # TODO: return next probs
        _next_probs = jnp.take_along_axis(probs, next_idx, axis=-1)

        return next_tokens

    def prefill(self, input_tokens: Array, state: InferenceState) -> InferenceState:
        logger.info(f"Compiling prefill for sequence length {input_tokens.shape[1]}")
        key, sample_key = jax.random.split(state.key)
        logits, out_cache = self.model.apply(
            state.params, x=input_tokens, sequence_lens=state.seq_lens, kv_cache=state.kv_cache
        )
        next_token = self.sample_logits(logits, sample_key)
        return InferenceState(
            next_token=next_token,
            kv_cache=out_cache,
            key=key,
            seq_lens=self.update_seq_lens(t=1, seq_lens=state.seq_lens),
            params=state.params,
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
        return InferenceState(
            next_token=self.sample_logits(logits, sample_key),
            kv_cache=out_cache,
            key=key,
            seq_lens=self.update_seq_lens(t=1, seq_lens=state.seq_lens),
            params=state.params,
        )

    def batch_decode(self, x: Array, seq_lens: Array, key: Array, params: PyTree) -> tuple[Array, PyTree]:
        B, T = x.shape
        inference_state = InferenceState(
            next_token=jnp.empty((self.batch_size, 1), dtype=jnp.int32),
            kv_cache=self.model.init_kv_cache(x, dtype=self.config.kv_cache_dtype),
            key=key,
            seq_lens=seq_lens,
            params=params,
        )
        out_tokens = x

        # TODO: use stax timer and use a blocking call
        start_time = time.perf_counter()

        precompiled_length = max(self.inital_sequence_len, self.compute_max_padding_length(seq_lens))
        inference_state = self.precompile_dict["prefill"][precompiled_length](x, inference_state)
        ttft_time = jnp.array(time.perf_counter() - start_time)

        out_tokens = jnp.concatenate((out_tokens, inference_state.next_token), axis=-1)
        tps = []

        attention_len: int = max(
            self.compute_attention_length(inference_state.kv_cache[0].length), self.inital_sequence_len
        )

        start_time = time.perf_counter()
        for _ in range(T + 1, self.max_seq_len):
            inference_state = self.precompile_dict["decode"][attention_len](inference_state)
            out_tokens = jnp.concatenate((out_tokens, inference_state.next_token), axis=-1)

            if inference_state.kv_cache[0].length >= attention_len:
                attention_len *= 2

            if _ % 10 == 0 and _ != 0:
                time_end = time.perf_counter()
                tps.append(B * 10 / (time_end - start_time))
                print(f"Stats: {tps[-1]:.2f} tokens/second")
                start_time = time.perf_counter()

        metrics = {"tokens_per_second": jnp.array(tps[1:]).mean() if tps else 0.0, "ttft": ttft_time}
        return out_tokens, metrics

    def multi_batch_decode(
        self, batch_tokens: Array, seq_lens: Array, key: Array, params: PyTree
    ) -> tuple[Array, PyTree]:
        B, _ = batch_tokens.shape
        batches = math.ceil(B / self.batch_size)
        output_tokens = jnp.array([], dtype=jnp.int32)
        metrics = []
        for i in range(batches):
            tokens = batch_tokens[i * self.batch_size : (i + 1) * self.batch_size]
            seq_lens_batch = seq_lens[i * self.batch_size : (i + 1) * self.batch_size]
            batch_output, batch_metrics = self.batch_decode(tokens, seq_lens_batch, key, params)
            output_tokens = jnp.concatenate((output_tokens, batch_output), axis=0)
            metrics.append(batch_metrics)

        metrics = jax.tree.map(lambda *x: jnp.stack(x).mean(axis=0), *metrics)
        return output_tokens, metrics

    def rollout(self, batch_tokens: Array, seq_lens: Array, key: Array, params: PyTree) -> tuple[Array, PyTree]:
        B, _ = batch_tokens.shape
        if B > self.batch_size:
            return self.multi_batch_decode(batch_tokens, seq_lens, key, params)
        return self.batch_decode(batch_tokens, seq_lens, key, params)

    def __call__(self, texts: list[str], key: Array, params: PyTree) -> tuple[list[str], PyTree]:
        inp_tokens, seq_lens = self.tokenize(texts)
        output_tokens, metrics = self.rollout(inp_tokens, seq_lens, key, params)
        output = self.detokenizer(output_tokens)
        return output, metrics


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
        temperature=0.6, top_p=0.95, top_k=50, max_seq_len=128, batch_size=1, group_size=1, kv_cache_dtype="bfloat16"
    )

    engine = InferenceEngine(model, params, config)

    key = jax.random.PRNGKey(2303)
    params = model.init_state(jax.random.PRNGKey(0), None, abstract=False)

    tokenizer_inp = ["Create a generating series for the composition of parts in {1, 2, \ldots, 100}"]

    inp_tokens, sequence_lens = engine.tokenize(tokenizer_inp)
    breakpoint()

    key = jax.random.PRNGKey(2303)
    output_tokens, metrics = engine.rollout(inp_tokens, sequence_lens, key, params)

    output = engine.detokenizer(output_tokens)
    print(metrics)
