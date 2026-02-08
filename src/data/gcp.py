import json
from typing import Iterable, Optional

import gcsfs
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import OmniMath, load_omni_math


def open_gcs(path: str, mode: str = "rb"):
    """Open a GCS object via `gcsfs`."""
    return gcsfs.GCSFileSystem().open(path, mode)


def write_jsonl(examples: Iterable[OmniMath], gs_path: str) -> None:
    """Write examples as JSONL to `gs://...`."""
    with open_gcs(gs_path, "wt") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_json(), ensure_ascii=False) + "\n")


@hydra.main(
    version_base=None,
    config_path=None,
    config_name=None,
)
def stage_omnimath_to_gcs(cfg: DictConfig) -> None:
    """
    Export Omni-MATH to a GCS JSONL file via Hydra config.
    Usage:
        python src/data/gcp.py gs_path=gs://bucket/path/omnimath.jsonl split=test ...
    """
    # Unpack config with defaults
    gs_path = cfg.get("gs_path")
    name = cfg.get("name", "KbsdJames/Omni-MATH")
    split = cfg.get("split", "test")
    cache_dir = cfg.get("cache_dir", None)
    min_difficulty = cfg.get("min_difficulty", None)
    max_difficulty = cfg.get("max_difficulty", None)
    domain_contains = cfg.get("domain_contains", None)
    source_contains = cfg.get("source_contains", None)

    if gs_path is None:
        raise ValueError("`gs_path` must be specified (e.g. gs_path=gs://bucket/data.jsonl).")

    examples = load_omni_math(
        name=name,
        split=split,
        cache_dir=cache_dir,
        min_difficulty=min_difficulty,
        max_difficulty=max_difficulty,
        domain_contains=domain_contains,
        source_contains=source_contains,
    )
    write_jsonl(examples, gs_path)


if __name__ == "__main__":
    stage_omnimath_to_gcs()
