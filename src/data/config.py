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
    def from_json(cls, data: dict):
        prompt = data.get("prompt")
        answer = data.get("answer")
        solution = data.get("solution")

        if prompt is None or answer is None:
            raise ValueError("Prompt and answer are required")

        return cls(prompt=prompt, answer=answer, solution=solution)

    def get_json(self) -> dict:
        object = {"prompt": self.prompt, "answer": self.answer, "solution": self.solution}
        return json.dumps(object)

@dataclass
class DataConfig:
    name: str
    batch_size: int
    gcs_path: str | None = None
    split: str = "train"
    sample: type[Sample] = Sample

@struct.dataclass
class RLBatch:
    tokens: jax.Array  # [batch_size, max_seq_len]
    token_mask: jax.Array  # [batch_size, max_seq_len], which tokens are LLM
    rewards: jax.Array  # [batch_size, ], reward at each token
