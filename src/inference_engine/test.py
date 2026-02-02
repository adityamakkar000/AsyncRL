import jax.numpy as jnp


def calculate_max_padding_length(self, seq_lens) -> int:
    return jnp.max(len(seq_lens))
