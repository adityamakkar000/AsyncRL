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
    reference_model_logprobs: jax.Array  # [batch_size, max_seq_len]
    seq_lens: jax.Array  # [batch_size, ] what is the length of each sequence
    rewards: jax.Array  # [batch_size, max_seq_len], reward at each token
    token_mask: jax.Array  # [batch_size, max_seq_len], which tokens to train on
