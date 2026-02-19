from dataclasses import dataclass
import json

import jax
from flax import struct


@dataclass
class Sample:
    prompt: str
    answer: str
    solution: str | None

    @classmethod
    def from_dict(cls, data: dict):

        prompt = data.get("problem")
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
    tokens: jax.Array  # [batch_size, max_seq_len]
    token_mask: jax.Array  # [batch_size, max_seq_len], which tokens are LLM
    rewards: jax.Array  # [batch_size, ], reward at each token
