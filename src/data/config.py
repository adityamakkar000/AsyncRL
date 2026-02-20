from dataclasses import dataclass

import jax
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
    name: str
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
    tokens: jax.Array  # [B, max_seq_len]
    reference_model_logprobs: jax.Array  # [B, max_seq_len]
    seq_lens: jax.Array  # [B] what is the length of each sequence to not include padding tokens
    rewards: jax.Array  # [B], reward at each token
    group_mean: jax.Array  # [B], mean reward for each group
    group_std: jax.Array  # [B], std reward for each group
    token_mask: jax.Array  # [B, max_seq_len], which tokens to train on
