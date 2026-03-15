from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from src.vllm_engine.config import vLLMConfig


@dataclass
class modelConfig:
    model_name: str = MISSING
    use_best_ckpt: bool = False
    step_number: Optional[int] = None


@dataclass
class evalConfig:
    tasks: list[str] = MISSING
    model_config: modelConfig = field(default_factory=modelConfig)
    vllm_config: vLLMConfig = field(default_factory=vLLMConfig)
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 8192
    debug: bool = False
    epochs: int = 4

    # not to include in hash
    max_connections: int = 100
