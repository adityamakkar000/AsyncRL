import einops
import jax
import jax.numpy as jnp
from flash_attention import SegmentIds, flash_attention
from flax import linen as nn


class NormalGQA(nn.Module):
    model_dim: int
    n_heads: int
    n_groups: int
    head_dim: int
    kv_group_size: int
    rope_base: int
    activation_dtype: jnp.dtype
    n_layers: int

    @nn.compact
    def __call__(self, x, mask):
        for _ in range(self.n_layers):
            B, T, C = x.shape

            q = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(x)
            k = nn.Dense(features=self.n_groups * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
            v = nn.Dense(features=self.n_groups * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)

            q = einops.rearrange(q, "... t (h d) -> ... t h d", d=self.head_dim)
            k = einops.rearrange(k, "... t (g d) -> ... t g d", d=self.head_dim)
            v = einops.rearrange(v, "... t (g d) -> ... t g d", d=self.head_dim)
            q = einops.rearrange(q, pattern="b t (g r) d -> b t g r d", g=k.shape[-2])

            wei = jnp.einsum("btgrd, bTgd -> btTgr", q.astype(jnp.float32), k.astype(jnp.float32)) * (
                self.head_dim**-0.5
            )

            wei = einops.rearrange(wei, pattern="b t T g r -> b (g r) t T ")
            wei = jnp.where(mask == 1, wei, -jnp.inf)
            wei = jax.nn.softmax(wei, axis=-1)
            nan_mask = ~jnp.isnan(wei)
            wei = jnp.where(nan_mask == 1, wei, 0)

            wei = einops.rearrange(tensor=wei, pattern="b (g r ) t T -> b t T g r", g=k.shape[2])

            out = jnp.einsum("btTgr, bTgd -> btgrd", wei, v.astype(jnp.float32))

            x = einops.rearrange(out, "b t g r d -> b t (g r d)")
        return x


class FlashGQA(nn.Module):
    model_dim: int
    n_heads: int
    n_groups: int
    head_dim: int
    kv_group_size: int
    rope_base: int
    activation_dtype: jnp.dtype
    n_layers: int

    @nn.compact
    def __call__(self, x, segment_ids: SegmentIds):
        for _ in range(self.n_layers):
            B, T, C = x.shape

            q = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.activation_dtype)(x)
            k = nn.Dense(features=self.n_groups * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)
            v = nn.Dense(features=self.n_groups * self.head_dim, use_bias=False, dtype=self.activation_dtype)(x)

            q = einops.rearrange(q, "... t (h d) -> ... h t d", d=self.head_dim)
            k = einops.rearrange(k, "... t (g d) -> ... g t d", d=self.head_dim)
            v = einops.rearrange(v, "... t (g d) -> ... g t d", d=self.head_dim)

            k = jnp.repeat(k, self.kv_group_size, axis=1)
            v = jnp.repeat(v, self.kv_group_size, axis=1)

            sm_scale = self.head_dim**-0.5
            out = flash_attention(q, k, v, sm_scale=sm_scale, causal=True, segment_ids=segment_ids)
            x = einops.rearrange(out, "b h t d -> b t (h d)")
        return x


def make_prompt_mask(max_seq_len: int, cache_len, seq_lens):
    """
    This function generates a boolean mask that identifies valid (non-padded) tokens
    within the cache region of each sequence. It handles left-padded sequences by
    masking out padding tokens at the beginning of each sequence.

        max_seq_len (int): The maximum sequence length including any tokens beyond the cache.
        cache_len (int): The length of the cached tokens (KV cache size).
        seq_lens (Array): Array of shape (batch_size,) containing the actual
            sequence lengths for each batch element. Each element should be <= cache_len.

    Returns:
        Array: A boolean mask of shape (batch_size, max_seq_len) where True indicates
            valid (non-padded) tokens within the cache region, and False indicates either
            padding tokens or positions beyond the cache.

    Example:
        >>> seq_lens = jnp.array([3, 5])
        >>> max_seq_len = 5
        >>> cache_len = 4
        >>> make_prompt_mask(max_seq_len, cache_len, seq_lens)
        # Returns:
        # [[False, False, True, True, False],
        #  [True,  True,  True, True, False]]
        #
        # First sequence: 3 valid tokens, left-padded with 1 token, 1 position beyond cache
        # Second sequence: 4 valid tokens (capped by cache_len), 1 position beyond cache
    """
    raw_length = jnp.arange(max_seq_len)[None, :]
    # left padding mask
    padding_mask = raw_length >= (cache_len - seq_lens[:, None])
    # cache mask
    cache_mask = raw_length < cache_len
    return padding_mask & cache_mask


model_dim = 1024
head_dim = 128
n_heads = 8
n_groups = 2
kv_group_size = 4
rope_base = 10_000
activation_dtype = jnp.float32
n_layers = 30

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (2, 128, model_dim))
seq_lens = jnp.array([32, 64])

# change to make
mask = jnp.tril(jnp.ones((128, 128), dtype=bool))[None, None, :, :]

normal_gqa = NormalGQA(model_dim, n_heads, n_groups, head_dim, kv_group_size, rope_base, activation_dtype, n_layers)
flash_gqa = FlashGQA(model_dim, n_heads, n_groups, head_dim, kv_group_size, rope_base, activation_dtype, n_layers)

init_key = jax.random.PRNGKey(42)

params = normal_gqa.init(init_key, x, mask)
flash_mask = make_prompt_mask(128, 128, seq_lens)


out1 = normal_gqa.apply(params, x, mask)
out2 = flash_gqa.apply(params, x, SegmentIds(flash_mask, flash_mask))  # shared params

breakpoint()
