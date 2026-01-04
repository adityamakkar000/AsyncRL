import jax
from jaxtyping import PyTree

from src.model import mainModel


class inferenceEngine:
    def __init__(self, model_module: mainModel):
        self.model_module = model_module

    def update_seq_lens(self, t: int, seq_lens: jax.Array):
        return t + seq_lens

    def precache_prefill(self):
        pass

    def compile_prefill(self):
        pass

    def prefill(
        self,
        params: PyTree,
        x: jax.Array,
        seq_lens: jax.Array,
        key: jax.Array,
    ):
        B, _ = x.shape

        initial_cache = self.model_module.init_kv_cache()

        out, cache = self.model_module.model(params, x, seq_lens, initial_cache)

        final_tokens = jax.random.categorical(key, out[:, -1, :], axis=-1)

        return final_tokens[:, None], cache, self.update_seq_lens(1, seq_lens)

    @jax.jit(static_argnums=(0,))
    def decode(
        self, state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        x, key, kv_cache, seq_lens, params = state

        logits, out_cache = self.model_module.model.apply(params, x, seq_lens, kv_cache)
        key, subkey = jax.random.split(key)

        next_tokens = jax.random.categorical(key, logits[:, -1, :], axis=-1)[:, None]

        seq_lens = self.update_seq_lens(1, seq_lens)

        return (next_tokens, subkey, out_cache, seq_lens, params)

    def rollout(self, chosen_model: str, max_tokens: int, x: jax.Array, seq_lens: jax.Array):
        # first init the model
        B, T = x.shape

        rng = jax.random.PRNGKey(0)
        params, optax_state, _ = self.model_module.init_state(chosen_model, rng)

        new_tokens, kv_cache, seq_lens = self.prefill(params, x, seq_lens, rng)

        for i in range(T, max_tokens):
            new_tokens, rng, kv_cache, seq_lens, params = self.decode_jit((new_tokens, rng, kv_cache, seq_lens, params))

        return new_tokens
