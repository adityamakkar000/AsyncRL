from dataclasses import dataclass

import jax
from flax import struct


@dataclass
class DataConfig:
    name: str
    batch_size: int = 4
    gcs_path: str | None = None
    prompt_column: str = "problem"

@struct.dataclass
class RLBatch:
    tokens: jax.Array  # [batch_size, max_seq_len]
    token_mask: jax.Array  # [batch_size, max_seq_len], which tokens are LLM
    rewards: jax.Array  # [batch_size, ], reward at each token
