import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from ...config import AttentionMask, KVCache, RopeMatrix
from ...utils import act_constrain
from .attention import GroupedQueryAttention
from .mlp import FeedForward
from .norm import RMSNorm


class Block(nn.Module):
    model_dim: int
    d_ff: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    activation_dtype: jnp.dtype = jnp.float32
    qk_norm: bool = False
    qk_norm_per_head: bool = True
    rms_eps: float = 1e-6
    post_norm: bool = False
    sliding_window: int | None = None
    full_attention_every: int = 1

    @nn.compact
    def __call__(
        self,
        carry: tuple[Array, KVCache | None],
        seq_lens: Array,
        masks: AttentionMask,
        rope_matrix: RopeMatrix,
        sliding_rope_matrix: RopeMatrix | None,
        layer: Array,
    ):
        x, layer_cache = carry
        shard = layer_cache is None

        # every layer runs this same code under nn.scan, so the layer type is picked from the layer index at runtime
        window = None
        if self.sliding_window is not None:
            assert sliding_rope_matrix is not None
            sliding = (layer + 1) % self.full_attention_every != 0
            rope_matrix = RopeMatrix(
                sin=jnp.where(sliding, sliding_rope_matrix.sin, rope_matrix.sin),
                cos=jnp.where(sliding, sliding_rope_matrix.cos, rope_matrix.cos),
            )
            # sequences that fit in the window see every token anyway, so no mask is needed
            max_attended = x.shape[1]
            if layer_cache is not None and x.shape[1] == 1:  # decode: one new token against the whole cache
                max_attended = layer_cache.max_len
            if max_attended > self.sliding_window:
                window = jnp.where(sliding, self.sliding_window, jnp.iinfo(jnp.int32).max)

        attention = GroupedQueryAttention(
            model_dim=self.model_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            activation_dtype=self.activation_dtype,
            qk_norm=self.qk_norm,
            qk_norm_per_head=self.qk_norm_per_head,
            rms_eps=self.rms_eps,
        )
        mlp = FeedForward(model_dim=self.model_dim, d_ff=self.d_ff, activation_dtype=self.activation_dtype)
        attention_norm = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)
        mlp_norm = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)
        attention_cache = None if layer_cache is None else (layer, layer_cache)

        x = act_constrain(x, shard)
        if self.post_norm:
            h, out_layer_cache = attention(x, masks, rope_matrix, attention_cache, window)
            x = act_constrain(x + attention_norm(h), shard)
            x = x + mlp_norm(mlp(x, shard))
        else:
            h, out_layer_cache = attention(attention_norm(x), masks, rope_matrix, attention_cache, window)
            x = act_constrain(x + h, shard)
            x = x + mlp(mlp_norm(x), shard)

        return (act_constrain(x, shard), out_layer_cache), None


RematBlock = nn.remat(Block)
