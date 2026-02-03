import math
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from transformers import AutoTokenizer

# from src.inference_engine import InferenceConfig
from src.model import KVCache, Model, ModelConfig, QwenConfig


@dataclass
class InferenceConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 50
    max_seq_len: int = 300
    batch_size: int = 4
    group_size: int = 1


class InferenceEngine:
    def __init__(self, model_module: Model, config: InferenceConfig):
        self.model_module = model_module
        self.model = self.model_module.model
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.batch_size = config.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_module.config.hf_model_name)

    def precompile(self) -> None:
        self.precompile_dict = {}
        curr_size = 2
        while curr_size <= self.max_seq_len:
            x_init = jnp.ones((1, curr_size), dtype=jnp.int32)
            seq_lens = jnp.array([curr_size])
            key = jax.random.PRNGKey(0)
            params = self.model_module.init_state(jax.random.PRNGKey(0), None, None)
            kv_cache = self.model_module.init_kv_cache(x_init)

            jit_func = jax.jit(self.prefill, static_argnums=(0,))
            self.precompile_dict[curr_size] = jit_func(params, x_init, seq_lens, key, kv_cache=kv_cache)
            curr_size *= 2

    def calculate_max_padding_length(self, seq_lens: list[list[int]]) -> int:
        max_token_len = 0
        for example in seq_lens:
            max_token_len = max(max_token_len, len(example))

        bit_length = max_token_len.bit_length()
        padded_length = 1 << bit_length
        return min(padded_length, self.max_seq_len)

    def tokenize(self, texts: list[str]) -> tuple[Array, Array]:
        inputs = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=True
            )
            for text in texts
        ]
        # padding_length = self.calculate_max_padding_length(inputs)
        seq_lens = jnp.array([len(x) for x in inputs], dtype=jnp.int32)
        # inputs = [(padding_length - len(x)) * [self.tokenizer.pad_token_id] + x for x in inputs]
        tokens = jnp.array(inputs)

        return tokens, seq_lens

    def detokenizer(self, tokens: Array) -> list[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=False)

    def update_seq_lens(self, t: int, seq_lens: jax.Array):
        return t + seq_lens

    def sample_logits(self, logits: Array, key: Array) -> Array:
        B, T, V = logits.shape
        logits = logits[:, -1, :] / self.config.temperature
        if self.config.top_k > 0:
            top_k = min(self.config.top_k, V)
            logits, base_indices = jax.lax.top_k(logits, top_k)
        else:
            base_indices = jnp.tile(jnp.arange(V), (B, 1))

        if self.config.top_p < 1.0:
            probs = jax.nn.softmax(logits, axis=-1)
            sort_idx = jnp.argsort(-probs, axis=-1)
            sorted_probs = jnp.take_along_axis(probs, sort_idx, axis=-1)
            sorted_logits = jnp.take_along_axis(logits, sort_idx, axis=-1)

            mask = jnp.cumsum(sorted_probs, axis=-1) <= self.config.top_p
            mask = mask.at[:, 0].set(True)

            filtered_logits = jnp.where(mask, sorted_logits, -jnp.inf)
            next_sorted_idx = jax.random.categorical(key, filtered_logits, axis=-1)[:, None]
            chosen_sorted = jnp.take_along_axis(sort_idx, next_sorted_idx, axis=-1)
            next_tokens = jnp.take_along_axis(base_indices, chosen_sorted, axis=-1)
        else:
            next_idx = jax.random.categorical(key, logits, axis=-1)[:, None]
            next_tokens = jnp.take_along_axis(base_indices, next_idx, axis=-1)

        return next_tokens

    def prefill(
        self, params: PyTree, x: Array, seq_lens: Array, key: Array, kv_cache: Optional[list[KVCache]] = None
    ) -> tuple[Array, list[KVCache], Array]:
        out, cache = self.model.apply(params, x=x, sequence_lens=seq_lens, kv_cache=kv_cache)
        return self.sample_logits(out, key), cache, self.update_seq_lens(1, seq_lens)

    @partial(jax.jit, static_argnums=(0,))
    def decode(self, state: tuple[Array, Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array, Array]:
        x, key, kv_cache, seq_lens, params = state

        key, sample_key = jax.random.split(key)
        logits, out_cache = self.model.apply(params, x=x, sequence_lens=seq_lens, kv_cache=kv_cache)
        next_tokens = self.sample_logits(logits, sample_key)
        seq_lens = self.update_seq_lens(t=1, seq_lens=seq_lens)
        return (next_tokens, key, out_cache, seq_lens, params)

    def batch_decode(self, x: Array, seq_lens: Array, key: Array, params: PyTree) -> Array:
        B, T = x.shape

        initial_cache = self.model_module.init_kv_cache(x)
        key, prefill_key = jax.random.split(key)
        next_tokens, kv_cache, seq_lens = self.prefill(params, x, seq_lens, prefill_key, kv_cache=initial_cache)

        tokens_output = jnp.concatenate((x, next_tokens), axis=-1)

        for _ in range(T, self.max_seq_len):
            start_time = time.perf_counter()
            next_tokens, key, kv_cache, seq_lens, params = self.decode((next_tokens, key, kv_cache, seq_lens, params))
            tokens_output = jnp.concatenate((tokens_output, next_tokens), axis=-1)
            time_end = time.perf_counter()
            tps = 1 / (time_end - start_time)
            print(f"Generated token {_} at {tps:.2f} tokens/second")

        return tokens_output

    def multi_batch_decode(self, batch_tokens: Array, seq_lens: Array, key: Array, params: PyTree) -> Array:
        B, _ = batch_tokens.shape
        batches = math.ceil(B / self.batch_size)
        output_tokens = jnp.array([], dtype=jnp.int32)
        for i in range(batches):
            tokens = batch_tokens[i * self.batch_size : (i + 1) * self.batch_size]
            seq_lens_batch = seq_lens[i * self.batch_size : (i + 1) * self.batch_size]
            batch_output = self.batch_decode(tokens, seq_lens_batch, key, params)
            output_tokens = jnp.concatenate((output_tokens, batch_output), axis=0)

        return output_tokens

    def rollout(self, batch_tokens: Array, seq_lens: Array, key: Array, params: PyTree) -> Array:
        B, _ = batch_tokens.shape
        if B > self.batch_size:
            return self.multi_batch_decode(batch_tokens, seq_lens, key, params)

        return self.batch_decode(batch_tokens, seq_lens, key, params)


if __name__ == "__main__":
    model_config = ModelConfig(
        "Qwen/Qwen3-0.6B",
        qwen_config=QwenConfig(
            vocab_size=151936,
            d_ff=3072,
            sequence_len=1024,
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
    config = InferenceConfig()
    engine = InferenceEngine(model, config)

    # inp = jnp.array([[0, 0, 1, 2, 3], [0, 0, 0, 2, 5], [3, 4, 5, 6, 9]], dtype=jnp.int32)
    # seq_lens = jnp.array([3, 2, 5], dtype=jnp.int32)

    tokenizer_inp = [
        "Create a generating series for the composition where each part is in 1 to 100. Put your answer in /boxed{}"
    ]
    inp_tokens, sequence_lens = engine.tokenize(tokenizer_inp)

    key = jax.random.PRNGKey(2303)
    params = model.init_state(jax.random.PRNGKey(0), None, abstract=False)
    output_tokens = engine.rollout(inp_tokens, sequence_lens, key, params)

    output = engine.detokenizer(output_tokens)
    print(output)
    breakpoint()
