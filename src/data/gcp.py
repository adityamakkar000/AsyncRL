import argparse
import gcsfs
import json
from typing import Iterable, Optional

from src.data.dataset import OmniMath, load_omni_math


def open_gcs(path: str, mode: str = "rb"):
    """Open a GCS object via `gcsfs`."""
    return gcsfs.GCSFileSystem().open(path, mode)


def write_jsonl(examples: Iterable[OmniMath], gs_path: str) -> None:
    """Write examples as JSONL to `gs://...`."""
    with open_gcs(gs_path, "wt") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_json(), ensure_ascii=False) + "\n")


def stage_omnimath_to_gcs(
    *,
    gs_path: str,
    name: str = "KbsdJames/Omni-MATH",
    split: str = "test",
    cache_dir: Optional[str] = None,
    min_difficulty: Optional[float] = None, # placeholders for now, aditya tell me how you want to do this
    max_difficulty: Optional[float] = None,
    domain_contains: Optional[str] = None,
    source_contains: Optional[str] = None,
) -> None:
    
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


def _main() -> None:
    parser = argparse.ArgumentParser(description="Dataset GCS utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    stage = sub.add_parser("stage-omnimath", help="Export Omni-MATH to a GCS JSONL file")
    stage.add_argument("--gs-path", required=True, help="Destination like gs://bucket/path/omnimath.jsonl")
    stage.add_argument("--split", default="test")
    stage.add_argument("--cache-dir", default=None)
    stage.add_argument("--min-difficulty", type=float, default=None)
    stage.add_argument("--max-difficulty", type=float, default=None)
    stage.add_argument("--domain-contains", default=None)
    stage.add_argument("--source-contains", default=None)

    args = parser.parse_args()

    if args.cmd == "stage-omnimath":
        stage_omnimath_to_gcs(
            gs_path=args.gs_path,
            split=args.split,
            cache_dir=args.cache_dir,
            min_difficulty=args.min_difficulty,
            max_difficulty=args.max_difficulty,
            domain_contains=args.domain_contains,
            source_contains=args.source_contains,
        )


if __name__ == "__main__":
    _main()
