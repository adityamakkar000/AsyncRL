
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Optional


@dataclass
class vLLMConfig:
    max_sequences: int = 128
    max_batched_tokens: int = 2048
    tensor_parallel_size: int = 1
    data_parallel_size: int = 8


@dataclass 
class modelConfig:
    model_name: str = MISSING
    use_best_ckpt: bool = False
    step_number: Optional[int] = None

    # def __post_init__(self):
    #     if self.use_best_ckpt and self.step_number is not None:
    #         raise ValueError("Cannot set both use_best_ckpt and step_number.")
    #     if not self.use_best_ckpt and self.step_number is None:
    #         raise ValueError("Must set either use_best_ckpt or step_number.")

@dataclass
class evalConfig:
    task: str = MISSING
    model_config: modelConfig = field(default_factory=modelConfig)
    vllm_config: vLLMConfig = field(default_factory=vLLMConfig)
    max_workers: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 8192
