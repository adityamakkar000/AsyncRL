from dataclasses import dataclass, field
from typing import Literal, Optional

from omegaconf import MISSING


@dataclass()
class SamplingConfig:
    """How to choose examples each call."""

    strategy: Literal["random"] = "random"
    seed: int = 0

    subset_fraction: float = 1.0
    min_difficulty: Optional[float] = None
    max_difficulty: Optional[float] = None
    domain_contains: Optional[str] = None
    source_contains: Optional[str] = None


@dataclass()
class GCSConfig:

    base_path: Optional[str] = None


@dataclass()
class OmniMathConfig:
    name: str = "KbsdJames/Omni-MATH"
    split: str = "test"
    cache_dir: Optional[str] = None


@dataclass()
class DataConfig:

    dataset: str = MISSING
    omni_math: OmniMathConfig = field(default_factory=OmniMathConfig)
    prompt_format: PromptFormatConfig = field(default_factory=PromptFormatConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    gcs: GCSConfig = field(default_factory=GCSConfig)

