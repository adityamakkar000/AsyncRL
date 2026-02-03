import dataclasses
import math
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
<<<<<<< HEAD
from transformers import AutoTokenizer
=======
>>>>>>> 10334d0 (working base inference)

# from src.inference_engine import InferenceConfig
from src.model import KVCache, Model, ModelConfig, QwenConfig


@dataclasses.dataclass
class InferenceConfig:
    temperature: float = 1.0
    top_p: float = 0.9
    max_seq_len: int = 50
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
        padding_length = self.calculate_max_padding_length(inputs)
        inputs = [(padding_length - len(x)) * [self.tokenizer.pad_token_id] + x for x in inputs]
        tokens = jnp.array(inputs)
        mask = tokens == self.tokenizer.pad_token_id
        seq_lens = jnp.sum(mask, axis=1)

        return tokens, seq_lens

    def detokenizer(self, tokens: Array) -> list[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def update_seq_lens(self, t: int, seq_lens: jax.Array):
        return t + seq_lens

    def prefill(
        self, params: PyTree, x: Array, seq_lens: Array, key: Array, kv_cache: Optional[list[KVCache]] = None
    ) -> tuple[Array, list[KVCache], Array]:
        B, _ = x.shape

        # TODO: sub int

        out, cache = self.model.apply(params, x=x, sequence_lens=seq_lens, kv_cache=kv_cache)

        final_tokens = jax.random.categorical(key, out[:, -1, :], axis=-1)

        return final_tokens[:, None], cache, self.update_seq_lens(1, seq_lens)

    @partial(jax.jit, static_argnums=(0,))
    def decode(self, state: tuple[Array, Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array, Array]:
        x, key, kv_cache, seq_lens, params = state

        logits, out_cache = self.model.apply(params, x=x, sequence_lens=seq_lens, kv_cache=kv_cache)
        key, subkey = jax.random.split(key)

        next_tokens = jax.random.categorical(subkey, logits[:, -1, :], axis=-1)[:, None]
        seq_lens = self.update_seq_lens(t=1, seq_lens=seq_lens)
        return (next_tokens, subkey, out_cache, seq_lens, params)

    def batch_decode(self, x: Array, seq_lens: Array, key: Array, params: PyTree) -> Array:
        B, T = x.shape
        initial_cache = self.model_module.init_kv_cache(x)
        next_tokens, kv_cache, seq_lens = self.prefill(params, x, seq_lens, key, kv_cache=initial_cache)

        tokens_ouput = jnp.concatenate((x, next_tokens), axis=-1)

        for t in range(T, self.max_seq_len):
            next_tokens, key, kv_cache, seq_lens, params = self.decode((next_tokens, key, kv_cache, seq_lens, params))
            tokens_ouput = jnp.concatenate((tokens_ouput, next_tokens), axis=-1)

        return tokens_ouput

    def multi_batch_decode(self, batch_tokens: Array, seq_lens: Array, key: Array, params: PyTree) -> Array:
        B, _ = batch_tokens.shape
        batches = math.ceil(B / self.batch_size)
        output_tokens = jnp.array([], dtype=jnp.int32)
        for i in range(batches):
            tokens = batch_tokens[i * self.batch_size : (i + 1) * self.batch_size]
            seq_lens_batch = seq_lens[i * self.batch_size : (i + 1) * self.batch_size]
            self.batch_decode(tokens, seq_lens_batch, key, params)
            output_tokens = jnp.concatenate((output_tokens, tokens), axis=0)

        return output_tokens

    def rollout(self, batch_tokens: Array, seq_lens: Array, key: Array, params: PyTree) -> Array:
        B, _ = batch_tokens.shape
        if B > self.batch_size:
            return self.multi_batch_decode(batch_tokens, seq_lens, key, params)

        return self.batch_decode(batch_tokens, seq_lens, key, params)


if __name__ == "__main__":
    vocab_size: int = 151936
    d_ff: int = 3072
    sequence_len: int = 20
    model_dim: int = 1024
    n_heads: int = 16
    n_groups: int = 8
    n_layers: int = 28
    head_dim: int = 128
    model_dtype: str = "float32"

    qwen_config = QwenConfig(
        vocab_size=vocab_size,
        d_ff=d_ff,
        sequence_len=sequence_len,
        model_dim=model_dim,
        n_heads=n_heads,
        n_groups=n_groups,
        head_dim=head_dim,
        n_layers=n_layers,
        rope_base=10_000,
        activation_dtype=model_dtype,
    )
    model_config = ModelConfig("Qwen/Qwen3-0.6B", qwen_config=qwen_config)
    model = Model(model_config)
    config = InferenceConfig()
    engine = InferenceEngine(model, config)

    # inp = jnp.array([[0, 0, 1, 2, 3], [0, 0, 0, 2, 5], [3, 4, 5, 6, 9]], dtype=jnp.int32)
    # seq_lens = jnp.array([3, 2, 5], dtype=jnp.int32)

    tokenizer_inp = ["Hello, how are you?"]
    inp_tokens, sequence_lens = engine.tokenize(tokenizer_inp)

    key = jax.random.PRNGKey(0)
    params = model.init_state(jax.random.PRNGKey(0), None, abstract=False)
    output_tokens = engine.rollout(inp_tokens, sequence_lens, key, params)

    output = engine.detokenizer(output_tokens)

    breakpoint()
