import jax
import jax.numpy as jnp


def convert_dtype(dtype_str: str) -> jnp.dtype:
    if dtype_str == "float32":
        return jnp.float32
    elif dtype_str == "bfloat16":
        return jnp.bfloat16
    elif dtype_str == "float16":
        return jnp.float16
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")


def make_attention_mask(t: int, T: int, seq_lens: jax.Array) -> jax.Array:
    """
    Create an attention mask for sequences with padding and causal masking.
    Args:
        t (int): Current time step or query length.
        T (int): Total sequence length or key length.
        seq_lens (jax.Array): Array of sequence lengths for each batch element.
    Returns:
        jax.Array: Attention mask of shape (batch_size, 1, t, T).

    example:
        seq_lens = jnp.array([3, 5])
        t = 3
        T = 5
        prompt_mask = [[0 0 1 1 1]
                       [1 1 1 1 1]]
        tril = [[[True True True False False],
                 [True True True True False],
                 [True True True True True]]]
        returns:
        [[[0 0 1 0 0]
          [0 0 1 1 0]
          [0 0 1 1 1]]

         [[1 1 1 0 0]
          [1 1 1 1 0]
          [1 1 1 1 1]]]
    """

    def create_prompt_mask(padding_len: int, seq_lens: jax.Array) -> jax.Array:
        """
        Create a prompt mask to handle left-padding in sequences.
        Args:
            padding_len (int): The total length of the sequences including padding.
            seq_lens (jax.Array): Array of sequence lengths for each batch element.
        Returns:
            jax.Array: Prompt mask of shape (batch_size, padding_len).

        example:
            seq_lens = jnp.array([3, 5])
            padding_len = 5
            [[0 1 2 3 4]] >= [[2], [0]]
            returns:
            [[0 0 1 1 1]
             [1 1 1 1 1]]
        """

        return jnp.arange(padding_len)[None, :] >= (padding_len - seq_lens[:, None])

    def make_tril_mask(t: int, T: int) -> jax.Array:
        """
        Create a lower triangular mask for causal attention.
        Args:
            t (int): Current time step or query length.
            T (int): Total sequence length or key length.
        Returns:
            jax.Array: Lower triangular mask of shape (t, T) [last row will always be True].

        example:
            t = 3
            T = 5

            T_arange = [[0 1 2 3 4]]
            query_position = [[0 1 2 3 4],
                              [0 1 2 3 4],
                              [0 1 2 3 4]]
            key_position  = [[ 0 0 0 0 0],
                             [ 1 1 1 1 1],
                             [ 2 2 2 2 2],
                             [ 3 3 3 3 3],
                             [ 4 4 4 4 4]]
            key_position[-t:] = [[2 2 2 2 2],
                                 [3 3 3 3 3],
                                 [4 4 4 4 4]]
            returns:
            [[True True True False False],
             [True True True True False],
             [True True True True True]]

        """

        T_arange = jnp.arange(T)[None, :]
        query_position = jnp.repeat(T_arange, t, axis=0)
        key_position = jnp.transpose(jnp.repeat(T_arange, T, axis=0))
        return query_position <= key_position[-t:, :]

    prompt_mask = create_prompt_mask(padding_len=T, seq_lens=seq_lens)
    tril = make_tril_mask(t, T)[None, None, :, :]
    return prompt_mask[:, None, None, :] * tril
