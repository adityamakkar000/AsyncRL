import math
import time
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from stax.logger import staxLogger as logger
from transformers import AutoTokenizer

from src.model import KVCache, Model, ModelConfig, QwenConfig

from .config import InferenceConfig

"""
#TODO: Inference 
- multihost 
- fix up static batching
- integrate dataclass for return outputs from prefill 
- precompile along batch and T for prefill
- precompile decode along batch 
- max sequence length + stop token breaking
- roll kv cache 
- donate kv cache memory optimization 
- setup inference loop
- integrate tokenizer into single call function
- return back prob tokens
- precompile attention length so no need to do full 16k for every turn
"""


class InferenceEngine:
    def __init__(self, model_module: Model, config: InferenceConfig):
        self.model_module = model_module
        self.model = self.model_module.model
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.batch_size = config.batch_size
        self.group_size = config.group_size
        self.num_prompts = self.batch_size // self.group_size
        self.precompile_t = 128

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_module.config.hf_model_name)

    def _precompile(self, params: PyTree) -> None:
        """Precompile prefill function for different sequence lengths up to max_seq_len.
        The structure self.precompiled_dict = dict[seq_len --> compiled_fn]."""
        self.precompile_dict = {}

        curr_seq_len = self.precompile_t
        while curr_seq_len <= self.max_seq_len:
            x_init = jnp.ones((self.num_prompts, curr_seq_len), dtype=jnp.int32)
            seq_lens = jnp.array([curr_seq_len] * self.num_prompts)
            key = jax.random.PRNGKey(0)
            kv_cache = self.model_module.init_kv_cache(x_init, dtype=self.config.kv_cache_dtype)

            self.precompile_dict[curr_seq_len] = jax.jit(self.prefill)
            logger.info(f"Precompiled prefill function for sequence length {curr_seq_len}")
            _output = self.precompile_dict[curr_seq_len](params, x_init, seq_lens, key, kv_cache=kv_cache)
            curr_seq_len *= 2

    def compute_max_power_of_two(self, n: int, upper_bound: int) -> int:
        return min(1 << (n.bit_length()), upper_bound)

    def calculate_max_padding_length(self, seq_lens: Array) -> int:
        return self.compute_max_power_of_two(max(seq_lens).item(), self.max_seq_len)

    def tokenize(self, texts: list[str]) -> tuple[Array, Array]:
        inputs = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=True
            )
            for text in texts
        ]
        seq_lens = jnp.array([len(x) for x in inputs], dtype=jnp.int32)
        padding_length = self.calculate_max_padding_length(seq_lens)
        inputs = [(padding_length - len(x)) * [self.tokenizer.pad_token_id] + x for x in inputs]
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

    def prefill(
        self, params: PyTree, x: Array, seq_lens: Array, key: Array, kv_cache: Optional[list[KVCache]] = None
    ) -> tuple[Array, list[KVCache], Array]:
        out, cache = self.model.apply(params, x=x, sequence_lens=seq_lens, kv_cache=kv_cache)
        return self.sample_logits(out, key), cache, self.update_seq_lens(1, seq_lens)

    def precompiled_prefill(
        self, params: PyTree, x: Array, seq_lens: Array, key: Array, kv_cache: Optional[list[KVCache]] = None
    ) -> tuple[Array, list[KVCache], Array]:
        breakpoint()
        precompiled_length = max(self.precompile_t, self.calculate_max_padding_length(seq_lens))

        prefill_func = self.precompile_dict[precompiled_length]
        breakpoint()
        return prefill_func(params, x=x, seq_lens=seq_lens, key=key, kv_cache=kv_cache)

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

        initial_cache = self.model_module.init_kv_cache(x, dtype=self.config.kv_cache_dtype)
        key, prefill_key = jax.random.split(key)
        next_tokens, kv_cache, seq_lens = self.precompiled_prefill(
            params, x, seq_lens, prefill_key, kv_cache=initial_cache
        )

        tokens_output = jnp.concatenate((x, next_tokens), axis=-1)

        start_time = time.perf_counter()
        for _ in range(T + 1, self.max_seq_len):
            next_tokens, key, kv_cache, seq_lens, params = self.decode((next_tokens, key, kv_cache, seq_lens, params))
            tokens_output = jnp.concatenate((tokens_output, next_tokens), axis=-1)
            if _ % 10 == 0 and _ != 0:
                time_end = time.perf_counter()
                tps = B * 10 / (time_end - start_time)
                print(f"Stats: {tps:.2f} tokens/second")
                start_time = time.perf_counter()

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


def get_memory_usage():
    stats = jax.local_devices()[0].memory_stats()
    return stats["bytes_in_use"] / (1024**3)


if __name__ == "__main__":
    print(f"Memory inital: {get_memory_usage()}")  # 6.48e-05 GB

    # ------------ init model --------------
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
            activation_dtype="bfloat16",
        ),
    )
    model = Model(model_config)
    config = InferenceConfig(
        temperature=0.6, top_p=0.95, top_k=50, max_seq_len=300, batch_size=256, group_size=32, kv_cache_dtype="bfloat16"
    )
    engine = InferenceEngine(model, config)

    key = jax.random.PRNGKey(2303)
    params = model.init_state(jax.random.PRNGKey(0), None, abstract=False)

    engine._precompile(params)
    breakpoint()

    jax.block_until_ready(params)
    print(f"Memory after params init: {get_memory_usage()}")  # 2.80 GB

    # ------------ tokenize inputs --------------
    tokenizer_inp = [
        "Create a generating series for the composition where each part is in 1 to 100. Put your answer in /boxed{}",
        "Explain the theory of relativity with manifolds and reimannian geometry.",
    ] * 8
    inp_tokens, sequence_lens = engine.tokenize(tokenizer_inp)
    breakpoint()

    # ------------ run inference --------------
    output_tokens = engine.rollout(inp_tokens, sequence_lens, key, params)
    output = engine.detokenizer(output_tokens)

    print(output)

    # ------------ memory analysis -----------
    print("Starting memory analysis...")

    kv_cache = engine.model_module.init_kv_cache(inp_tokens, dtype=engine.config.kv_cache_dtype)

    compiled_step = engine.decode.trace((inp_tokens, key, kv_cache, sequence_lens, params)).lower().compile()
    compiled_stats = compiled_step.memory_analysis()
    total = (
        compiled_stats.temp_size_in_bytes
        + compiled_stats.argument_size_in_bytes
        + compiled_stats.output_size_in_bytes
        - compiled_stats.alias_size_in_bytes
    )
    print(f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**3):.2f} GB")
    print(f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**3):.2f} GB")
    print(f"Output size: {compiled_stats.output_size_in_bytes / (1024**3):.2f} GB")
    print(f"Total size: {total / (1024**3):.2f} GB")
    breakpoint()
