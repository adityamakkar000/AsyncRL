from dataclasses import dataclass, field
from typing import Any, cast

import jax
import numpy as np
from flax import struct
from omegaconf import MISSING


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
class DatasetConfig:
    name: str = MISSING
    verifier: dict[str, Any] = MISSING
    filters: list[dict[str, Any]] = field(default_factory=list)


@struct.dataclass
class RLBatch:
    tokens: jax.Array  # [B, max_seq_len] where B = P * G, P = num prompts, G = group size
    reference_model_logprobs: jax.Array  # [B, max_seq_len] logprobs of sampled token
    seq_lens: jax.Array  # [B] length of sequences (excluding padding)
    rewards: jax.Array  # [B] reward of each sequence
    group_mean: jax.Array  # [B] mean reward for sequence's group
    group_std: jax.Array  # [B] std of reward for sequence's group
    token_mask: jax.Array  # [B, T] mask for applying rl

    @classmethod
    def get_test_batch(cls, batch_size: int, max_seq_len: int) -> "RLBatch":
        return cls.from_numpy(
            tokens=np.zeros((batch_size, max_seq_len), dtype=np.int32),
            reference_model_logprobs=np.zeros((batch_size, max_seq_len), dtype=np.float32),
            seq_lens=np.zeros((batch_size,), dtype=np.int32),
            rewards=np.zeros((batch_size,), dtype=np.float32),
            group_mean=np.zeros((batch_size,), dtype=np.float32),
            group_std=np.ones((batch_size,), dtype=np.float32),
            token_mask=np.zeros((batch_size, max_seq_len), dtype=np.bool),
        )

    @classmethod
    def from_numpy(
        cls,
        tokens: np.ndarray,
        reference_model_logprobs: np.ndarray,
        seq_lens: np.ndarray,
        rewards: np.ndarray,
        group_mean: np.ndarray,
        group_std: np.ndarray,
        token_mask: np.ndarray,
    ) -> "RLBatch":
        return cls(
            **jax.tree.map(
                lambda x: cast(jax.Array, x),
                {
                    "tokens": tokens,
                    "reference_model_logprobs": reference_model_logprobs,
                    "seq_lens": seq_lens,
                    "rewards": rewards,
                    "group_mean": group_mean,
                    "group_std": group_std,
                    "token_mask": token_mask,
                },
            )
        )


@dataclass
class InferenceRollout:
    sample: Sample
    rollout_strs: list[str]
    rollout_tokens: list[np.ndarray]
    rollout_logprobs: list[np.ndarray]
    weight_iteration: list[int]
    n_rollouts: int = 0
    is_eval: bool = False

    def __len__(self):
        assert len(self.rollout_logprobs) == len(self.rollout_tokens), (
            "rollout_logprobs and rollout_tokens must have the same length"
        )
        return len(self.rollout_tokens)

    @property
    def lag(self) -> int:
        return min(self.weight_iteration) if self.weight_iteration else 0
