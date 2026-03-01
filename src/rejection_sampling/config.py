from dataclasses import dataclass, field

from omegaconf import MISSING

from src.vllm_engine.config import vLLMConfig


@dataclass
class rejectionSamplingConfig:
    datasets: list[str] = MISSING
    vllm_config: vLLMConfig = field(default_factory=vLLMConfig)
    num_samples: int = 10
