import json
import os

import gcsfs
from datasets import Dataset

from src.data.config import Sample


def samples_to_dataset(samples: list[Sample]) -> Dataset:
    """Convert a list of Sample to a HuggingFace Dataset (prompt, answer, solution columns)."""
    return Dataset.from_list([s.get_dict() for s in samples])


def load_jsonl_from_gcs(gs_prefix: str) -> list[dict]:
    """Load all JSONL files under gs_prefix (e.g. gs://bucket/data/omnimath/), return list of rows."""
    fs = gcsfs.GCSFileSystem()
    prefix = gs_prefix.replace("gs://", "") if gs_prefix.startswith("gs://") else gs_prefix
    if not prefix.endswith("/"):
        prefix += "/"

    rows: list[dict] = []
    for path in fs.ls(prefix):
        if not path.endswith(".jsonl"):
            continue
        with fs.open(path, "r") as f:
            for line in f:
                rows.append(json.loads(line.strip()))
    return rows


def upload_local_file_to_gcs(local_path: str, gs_path: str):
    """Upload a local file to GCS. gs_path should be like 'gs://bucket/path/file.jsonl'."""
    fs = gcsfs.GCSFileSystem()
    gcs_path = gs_path.replace("gs://", "") if gs_path.startswith("gs://") else gs_path

    with open(local_path, "rb") as src:
        data = src.read()
        with fs.open(gcs_path, "wb") as dst:
            dst.write(data)


def write_dataset_to_local_jsonl(dataset: Dataset, local_path: str) -> None:
    """Write dataset to a local JSONL file (all columns)."""

    dataset.to_json(local_path, lines=True)


def delete_local_file(local_path: str) -> None:
    """Remove the local file to free disk. Make sure to only remove .jsonl files"""
    if os.path.isfile(local_path) and local_path.endswith(".jsonl"):
        os.remove(local_path)
