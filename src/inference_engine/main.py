import jax
from jaxtyping import PyTree

from src.model import mainModel


# inference class should
# prefill
# decode
class inferenceEngine:
    def __init__(self, model_module: mainModel):
        self.model_module = model_module

    def prefill(
        self,
        params: PyTree,
        x: jax.Array,
        seq_lens: jax.Array,
        padding_len: int,
        max_sequence_len: int,
        n_groups: int,
        head_dim: int,
        n_layers: int,
        key: jax.Array,
    ):
        B, _ = x.shape

        initial_cache = self.model.init_kv_cache()

        out, cache = self.model_module(params, x, seq_lens, initial_cache)

        final_tokens = jax.random.categorical(key, out[:, -1, :], axis=-1)

        return final_tokens

        # return final_tokens[:, None], cache, update_seq_lens(1, seq_lens)
