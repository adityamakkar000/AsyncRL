from dataclasses import dataclass

import jax
from flax import struct


# TODO: get a real dataclass
@dataclass
class DataConfig:
    name: str


@struct.dataclass
class SFTBatch:
    tokens: jax.Array  # [batch_size, max_seq_len]
    token_mask: jax.Array  # [batch_size, max_seq_len], which tokens are LLM


@struct.dataclass
class RLBatch:
    tokens: jax.Array  # [batch_size, max_seq_len]
    seq_lens: jax.Array  # [batch_size, ] what is the length of each sequence 
    token_train_lens: jax.Array  # [batch_size, ] which tokens are LLM completion
    rewards: jax.Array  # [batch_size, ], reward at each token
