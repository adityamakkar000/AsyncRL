from dataclasses import dataclass, field

from omegaconf import MISSING

from src.vllm_engine.config import vLLMConfig


@dataclass
class RejectionSingleSample:
    prompt: str
    answer: str
    solution: str | None
    pass_score: float

    @classmethod
    def from_dict(cls, data: dict):
        prompt = data.get("prompt")
        answer = data.get("answer")
        solution = data.get("solution")
        pass_score = data.get("pass_score")
        if prompt is None or answer is None or pass_score is None:
            raise ValueError("Prompt, answer and pass_score are required")

        return cls(prompt=prompt, answer=answer, solution=solution, pass_score=pass_score)

    def get_dict(self) -> dict:
        return {"prompt": self.prompt, "answer": self.answer, "solution": self.solution, "pass_score": self.pass_score}


@dataclass
class rejectionSamplingConfig:
    datasets: list[str] = MISSING
    gcs_paths: list[str] = MISSING
    vllm_config: vLLMConfig = field(default_factory=vLLMConfig)
    num_samples: int = -1
    pass_at: int = MISSING
    max_workers: int = MISSING
    temperature: float = MISSING
    top_p: float = MISSING
    hf_model_name: str = MISSING
    max_sequence_len: int = MISSING
