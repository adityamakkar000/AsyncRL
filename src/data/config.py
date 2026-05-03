from dataclasses import dataclass

import jax
import numpy as np
from flax import struct
from jaxtyping import Array


@dataclass
class Sample:
    prompt: str
    answer: str
    solution: str | None

    @classmethod
    def from_dict(cls, data: dict):
        prompt = data.get("prompt")
        answer = data.get("answer")
        solution = data.get("solution")

        if prompt is None or answer is None:
            raise ValueError("Prompt and answer are required")

        return cls(prompt=prompt, answer=answer, solution=solution)

    def get_dict(self) -> dict:
        return {"prompt": self.prompt, "answer": self.answer, "solution": self.solution}


@dataclass
class ProcessDatasetConfig:
    name: list[str]
    chunk_size: int
    seed: int


@dataclass
class DatasetConfig:
    name: str
    batch_size: int
    gcs_path: str | None = None


@dataclass
class DataConfig:
    train_config: DatasetConfig
    val_config: DatasetConfig


@struct.dataclass
class RLBatch:
    tokens: jax.Array | np.ndarray  # [B, max_seq_len] where B = P * G, P = num prompts, G = group size
    reference_model_logprobs: jax.Array | np.ndarray  # [B, max_seq_len] logprobs of sampled token
    seq_lens: jax.Array | np.ndarray  # [B] length of sequences (excluding padding)
    rewards: jax.Array | np.ndarray  # [B] reward of each sequence
    group_mean: jax.Array | np.ndarray  # [B] mean reward for sequence's group
    group_std: jax.Array | np.ndarray  # [B] std of reward for sequence's group

    @classmethod
    def get_test_batch(cls, batch_size: int, max_seq_len: int) -> "RLBatch":
        return cls(
            tokens=np.zeros((batch_size, max_seq_len), dtype=np.int32),
            reference_model_logprobs=np.zeros((batch_size, max_seq_len), dtype=np.float32),
            seq_lens=np.zeros((batch_size,), dtype=np.int32),
            rewards=np.zeros((batch_size,), dtype=np.float32),
            group_mean=np.zeros((batch_size,), dtype=np.float32),
            group_std=np.ones((batch_size,), dtype=np.float32),
        )

@dataclass
class InferenceRollout:
    sample: Sample
    rollout_strs: list[str]
    rollouts_tokens: list[Array]
    rollout_logprobs: list[Array]