import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array

from ...config import AttentionMask, BaseModel, KVCache, RopeMatrix
from .block import RematBlock
from .norm import RMSNorm
from .rope import RopeCorrection, gather_rope, no_correction, rope_tables


class Transformer(BaseModel):
    vocab_size: int
    sequence_len: int
    model_dim: int
    d_ff: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_layers: int
    rope_base: int
    activation_dtype: jnp.dtype = jnp.float32
    tie_weights: bool = False
    qk_norm: bool = False
    qk_norm_per_head: bool = True
    rms_eps: float = 1e-6
    post_norm: bool = False
    rope_correction: RopeCorrection = no_correction
    rope_scale: float = 1.0
    # With a sliding window, every `full_attention_every`-th layer attends to the whole sequence and uses
    # `rope_correction` and `rope_scale`. The other layers attend to the last `sliding_window` tokens and use plain RoPE.
    # This is overfitted to match Olmo for now
    sliding_window: int | None = None
    full_attention_every: int = 1

    @nn.compact
    def __call__(
        self,
        x: Array,  # [B, T]
        sequence_lens: jax.Array,  # [B, ]
        kv_cache: KVCache | None = None,
        apply_lm_head: bool = True,
    ) -> tuple[Array, KVCache | None]:
        B, T = x.shape

        embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dim,
            dtype=jnp.float32,
            name="token_emb",
        )
        x = embed_layer(x)

        t_start = kv_cache.length if kv_cache else jnp.zeros((B,), dtype=jnp.int32)

        pos = jnp.arange(T)[None, :]
        shifted_pos = pos + t_start[:, None]
        masks = AttentionMask(
            q_segment_ids=shifted_pos < sequence_lens[:, None],
            k_segment_ids=pos < sequence_lens[:, None],
        )

        def rope(correction: RopeCorrection, scale: float) -> RopeMatrix:
            sin_table, cos_table = rope_tables(
                self.sequence_len, self.head_dim, self.rope_base, correction, dtype=self.activation_dtype, scale=scale
            )
            sin, cos = gather_rope(sin_table, cos_table, shifted_pos)
            return RopeMatrix(sin=sin, cos=cos)

        rope_matrix = rope(self.rope_correction, self.rope_scale)
        sliding_rope_matrix = rope(no_correction, 1.0) if self.sliding_window is not None else None

        ScanBlock = nn.scan(
            RematBlock,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, 0),
            length=self.n_layers,
            unroll=self.n_layers if kv_cache is not None else 1,
        )
        (x, out_cache), _ = ScanBlock(
            model_dim=self.model_dim,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            activation_dtype=self.activation_dtype,
            qk_norm=self.qk_norm,
            qk_norm_per_head=self.qk_norm_per_head,
            rms_eps=self.rms_eps,
            post_norm=self.post_norm,
            sliding_window=self.sliding_window,
            full_attention_every=self.full_attention_every,
            name="Block",
        )(
            (x, kv_cache),
            sequence_lens,
            masks,
            rope_matrix,
            sliding_rope_matrix,
            jnp.arange(self.n_layers, dtype=jnp.int32),
        )

        if out_cache is not None:
            assert kv_cache is not None
            out_cache = out_cache.replace(length=kv_cache.length + 1 if T == 1 else sequence_lens)

        x = RMSNorm(activation_dtype=self.activation_dtype, eps=self.rms_eps)(x)

        if apply_lm_head:
            if self.tie_weights:
                x = embed_layer.attend(x)
            else:
                x = nn.Dense(features=self.vocab_size, use_bias=False, dtype=jnp.float32)(x)

        return x.astype(jnp.float32), out_cache

    @property
    def seq_len(self) -> int:
        return self.sequence_len

    @property
    def kv_shape(self):
        return (self.n_layers, self.n_kv_heads, self.head_dim)
