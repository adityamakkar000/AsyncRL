from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class InferenceConfig:
    temperature: float = MISSING
    top_k: int = MISSING
    top_p: float = MISSING
    max_seq_len: int = MISSING
    batch_size: int = MISSING
    group_size: int = MISSING
    kv_cache_dtype: str = MISSING
