from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NamedTuple, cast

import jax
import numpy as np
from flax import struct
from omegaconf import MISSING


class DataEvent(StrEnum):
    FINISHED = "finished"
    CONSUMED = "consumed"
    DROPPED = "dropped"


class SampleKey(NamedTuple):
    env: str
    epoch: int
    id: int


class ReplayKey(NamedTuple):
    key: SampleKey
    retries: int


@dataclass(frozen=True)
class Sample:
    env: str
    epoch: int
    id: int
    system_prompt: str | None
    prompt: str

    @property
    def key(self) -> SampleKey:
        return SampleKey(self.env, self.epoch, self.id)


@dataclass(frozen=True)
class Score:
    reward: float
    status: str
    extracted: str | None = None
    expected: str | None = None


@dataclass
class DatasetConfig:
    filters: list[Any] = field(default_factory=list)
    transforms: list[Any] = field(default_factory=list)


@dataclass
class EnvConfig:
    _target_: str = MISSING
    datasets: dict[str, DatasetConfig] = MISSING
    filters: list[Any] = field(default_factory=list)
    transforms: list[Any] = field(default_factory=list)
    use_system_prompt: bool = False


class RLBatch(struct.PyTreeNode):
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
    n_rollouts: int = 1
    is_eval: bool = False

    def __len__(self):
        assert len(self.rollout_logprobs) == len(self.rollout_tokens), (
            "rollout_logprobs and rollout_tokens must have the same length"
        )
        return len(self.rollout_tokens)

    @property
    def lag(self) -> int:
        return min(self.weight_iteration) if self.weight_iteration else 0


@dataclass
class ScoredRollout:
    rollout: InferenceRollout
    scores: list[Score]

    @property
    def rewards(self) -> np.ndarray:
        return np.array([score.reward for score in self.scores], dtype=np.float32)

    @property
    def zero_variance(self) -> bool:
        return len({score.reward for score in self.scores}) == 1
