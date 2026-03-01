from dataclasses import dataclass


@dataclass
class vLLMConfig:
    max_sequences: int = 128
    max_batched_tokens: int | str = 2048
    tensor_parallel_size: int = 2
    data_parallel_size: int = 4
