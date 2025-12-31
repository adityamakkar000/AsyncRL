import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from jaxtyping import Array

jax.config.update("jax_default_matmul_precision", "highest")
jax.numpy.set_printoptions(precision=9)


@struct.dataclass
class KVCache:
    k: Array
    v: Array
    length: int


def update_seq_lens(t: int, seq_lens: jax.Array) -> jax.Array:
    return t + seq_lens


def create_prompt_mask(padding_len: int, seq_lens: jax.Array) -> jax.Array:
    prompt_mask = jnp.arange(padding_len)[None, :] >= (padding_len - seq_lens[:, None])
    return prompt_mask


def make_tril_mask(t: int, T: int) -> jax.Array:
    query_position = jnp.stack([jnp.arange(T)] * t, axis=0)
    key_position = jnp.transpose(jnp.stack([jnp.arange(T)] * T, axis=0))[-t:]
    return query_position <= key_position


def make_attention_mask(t: int, T: int, seq_lens: jax.Array) -> jax.Array:
    prompt_mask = create_prompt_mask(padding_len=T, seq_lens=seq_lens)
    tril = make_tril_mask(t, T)[None, None, :, :]  # 1,1, t, T
    return prompt_mask[:, None, None, :] * tril


class FeedForward(nn.Module):
    d_ff: int
    model_dim: int
    model_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array):
        x_fc1 = nn.Dense(features=self.d_ff, use_bias=False, dtype=self.model_dtype)(x)
        x_fc2 = nn.Dense(features=self.d_ff, use_bias=False, dtype=self.model_dtype)(x)
        x = nn.silu(x_fc1) * x_fc2
        x_fc3 = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.model_dtype)(x)

        return x_fc3


class RMSNorm(nn.Module):
    model_dtype: jnp.dtype = jnp.float32
    shift: bool = False

    @nn.compact
    def __call__(self, x: Array):
        eps = 1e-6
        x /= jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)

        gamma = self.param("gamma", nn.initializers.ones, (x.shape[-1]), self.model_dtype)
        beta = self.param("beta", nn.initializers.ones, (x.shape[-1]), self.model_dtype) if self.shift else None

        x = x * gamma

        if self.shift:
            x += beta

        return x


class RoPE(nn.Module):
    sequence_len: int
    model_dim: int
    model_dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.model_dim % 2 == 0, "Hidden dimension must be even"

        m = jnp.arange(0, self.sequence_len, dtype=self.model_dtype)  # [0, 1, ..., t-1] , 1 x t
        pos = jnp.arange(0, self.model_dim, 2, dtype=self.model_dtype) / self.model_dim  # [0, 2, 4, ... d-2] 1 x C/2
        theta = 1.0 / (1000000**pos)  # 1 x C/2

        inp = jnp.einsum("t,k->tk", m, theta)  # t x c/2

        self.sin = jnp.sin(inp)  # t x c/2
        self.cos = jnp.cos(inp)  # t x c/2

    def __call__(self, x: Array, t_start: int):
        B, h, T, C = x.shape

        cos = jax.lax.dynamic_slice(self.cos, (t_start, 0), (T, self.cos.shape[-1]))[None, None]
        sin = jax.lax.dynamic_slice(self.sin, (t_start, 0), (T, self.sin.shape[-1]))[None, None]

        # cos = self.cos[t_start : t_start + T, :][None, None]  # 1, 1, t, c/2
        # sin = self.sin[t_start : t_start + T, :][None, None]  # 1, 1, t, c/2

        x1, x2 = x[..., : C // 2], x[..., C // 2 :]  # B h T c/2

        out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)  # B, h, t, C
        return out


class GroupedQueryAttention(nn.Module):
    model_dim: int
    n_heads: int
    n_groups: int
    q_norm: bool
    k_norm: bool
    max_sequence_len: int
    head_dim: int = 64
    model_dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.n_heads % self.n_groups == 0, "Number of heads must be divisible by number of kv groups"
        assert self.model_dim % self.n_heads == 0, "Model dim must be divisible by number of heads"

        self.kv_group_size = self.n_heads // self.n_groups  # number of heads in each kv group
        self.d_out = self.n_heads * self.head_dim

    @nn.compact
    def __call__(self, x: Array, seq_lens: jax.Array, kv_cache: KVCache, layer_ind: int):
        B, T, C = x.shape
        t_start = kv_cache.length

        # KV = [B, g, T, d]
        k_cache, v_cache = kv_cache.k, kv_cache.v

        q = nn.Dense(features=self.d_out, use_bias=False, dtype=self.model_dtype)(x)  # [B, t, h * d]
        k = nn.Dense(
            features=self.n_groups * self.head_dim,
            use_bias=False,
            dtype=self.model_dtype,
        )(x)  # [B, t, 2 * n_groups * head_dim]
        v = nn.Dense(
            features=self.n_groups * self.head_dim,
            use_bias=False,
            dtype=self.model_dtype,
        )(x)

        q = einops.rearrange(q, "... t (h d) -> ... h t d", d=self.head_dim)  # [B, h, t, d]
        k = einops.rearrange(k, "... t (g d) -> ... g t d", d=self.head_dim)  # [B, g, t, d]
        v = einops.rearrange(v, "... t (g d) -> ... g t d", d=self.head_dim)  # [B, g, t, d]

        if self.q_norm:
            q = RMSNorm(self.model_dtype)(q)

        if self.k_norm:
            k = RMSNorm(self.model_dtype)(k)

        queries = RoPE(self.max_sequence_len, q.shape[-1], self.model_dtype)(q, t_start)
        k = RoPE(self.max_sequence_len, k.shape[-1], self.model_dtype)(k, t_start)

        print(f"start: {t_start}, T: {T}")

        k, v = jax.tree.map(
            lambda cache, val: jax.lax.dynamic_update_slice_in_dim(cache, val.astype(cache.dtype), t_start, axis=2),
            (k_cache, v_cache),
            (k, v),
        )
        kv_cache = KVCache(k=k_cache, v=v_cache, length=t_start + T)

        keys = einops.repeat(k, "b g t d -> b (g r) t d", r=self.kv_group_size)
        values = einops.repeat(v, "b g t d -> b (g r) t d", r=self.kv_group_size)

        print(f"q: {queries.shape}, k: {keys.shape}, x: {x.shape}")

        wei = jnp.einsum(
            "...td, ...Td -> ...tT",
            queries.astype(jnp.float32),
            keys.astype(jnp.float32),
        ) / jnp.sqrt(self.head_dim)

        attention_mask = make_attention_mask(t=T, T=k.shape[-2], seq_lens=seq_lens)
        wei = jnp.where(attention_mask == 1, wei, -jnp.inf)
        wei = jax.nn.softmax(wei, axis=-1)
        nan_mask = ~jnp.isnan(wei)
        wei = jnp.where(nan_mask == 1, wei, 0)

        out = jnp.einsum("...htT, ...hTd -> ...htd", wei, values.astype(jnp.float32))
        out = einops.rearrange(out, "... h t d -> ... t (h d)")
        out = nn.Dense(features=self.model_dim, use_bias=False, dtype=self.model_dtype)(out)

        return out, kv_cache


class Block(nn.Module):
    d_ff: int
    sequence_len: int
    model_dim: int
    n_heads: int
    n_groups: int
    head_dim: int
    model_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, seq_lens: jax.Array, layer_cache: KVCache, layer_ind: int):
        connection_1 = x
        x = RMSNorm(model_dtype=self.model_dtype)(x)
        x, out_layer_cache = GroupedQueryAttention(
            model_dim=self.model_dim,
            n_heads=self.n_heads,
            n_groups=self.n_groups,
            q_norm=True,
            k_norm=True,
            max_sequence_len=self.sequence_len,
            head_dim=self.head_dim,
            model_dtype=self.model_dtype,
        )(x, seq_lens, layer_cache, layer_ind)

        x = x + connection_1

        connection_2 = x
        x = RMSNorm(model_dtype=self.model_dtype)(x)
        x = FeedForward(d_ff=self.d_ff, model_dim=self.model_dim, model_dtype=self.model_dtype)(x)
        x = x + connection_2

        return x, out_layer_cache


class Qwen3(nn.Module):
    vocab_size: int
    d_ff: int
    sequence_len: int
    model_dim: int
    n_heads: int
    n_groups: int
    head_dim: int
    n_layers: int
    model_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, seq_lens: jax.Array, kv_cache: list[KVCache]):
        embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dim,
            dtype=self.model_dtype,
            name="token_emb",
        )

        x = embed_layer(x)

        out_cache = []
        for i in range(self.n_layers):
            in_layer_cache = kv_cache[i]
            x, out_layer_cache = Block(
                d_ff=self.d_ff,
                sequence_len=self.sequence_len,
                model_dim=self.model_dim,
                n_heads=self.n_heads,
                n_groups=self.n_groups,
                head_dim=self.head_dim,
                model_dtype=self.model_dtype,
            )(x, seq_lens, in_layer_cache, layer_ind=i)

            out_cache.append(out_layer_cache)

        x = RMSNorm(model_dtype=self.model_dtype)(x)

        # logits = embed_layer.attend(x)
        logits = nn.Dense(features=self.vocab_size, use_bias=False, dtype=self.model_dtype)(x)
        return logits, kv_cache
