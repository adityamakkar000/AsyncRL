from dataclasses import dataclass, field

from omegaconf import MISSING

from src.vllm_engine.config import vLLMConfig


@dataclass
class modelConfig:
    model_name: str = MISSING
    step_number: int | None = None


@dataclass
class evalConfig:
    tasks: list[str] = MISSING
    model_config: modelConfig = field(default_factory=modelConfig)
    vllm_config: vLLMConfig = field(default_factory=vLLMConfig)
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 8192
    debug: bool = False
    epochs: int = 2
    max_concurrent_requests: int = 500
