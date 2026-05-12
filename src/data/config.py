from dataclasses import dataclass

import jax
import numpy as np
from flax import struct


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
    rollout_tokens: list[np.ndarray]
    rollout_logprobs: list[np.ndarray]
    weight_iteration: int

    def __len__(self):
        assert len(self.rollout_logprobs) == len(self.rollout_tokens), (
            "rollout_logprobs and rollout_tokens must have the same length"
        )
        return len(self.rollout_tokens)

    def update_weight_iteration(self, weight_iteration: int):
        if weight_iteration < 0:
            self.weight_iteration = weight_iteration
        else:
            self.weight_iteration = min(self.weight_iteration, weight_iteration)
