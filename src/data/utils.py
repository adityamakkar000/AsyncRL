import json
import os
from typing import Dict

import gcsfs
import numpy as np
from datasets import Dataset

from .config import RLBatch, Sample


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
    if fs.exists(prefix):
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


def compute_aux_metrics(batch: RLBatch) -> Dict[str, float]:
    token_mask = batch.reference_model_logprobs != -np.inf
    return {
        "mean_reward": np.mean(batch.rewards).item(),
        "std_reward": np.std(batch.rewards).item(),
        "max_reward": np.max(batch.rewards).item(),
        "min_reward": np.min(batch.rewards).item(),
        "mean_length": np.mean(token_mask.sum(axis=1)).item(),
        "median_length": np.median(token_mask.sum(axis=1)).item(),
    }


def filter_rejection_sampled_data(
    gcs_path: str, lower_bound: float | None = None, upper_bound: float | None = None
) -> list[dict]:
    assert lower_bound is not None or upper_bound is not None, (
        "At least one of lower_bound or upper_bound must be provided"
    )

    rows = load_jsonl_from_gcs(gcs_path)
    if lower_bound is not None:
        rows = [row for row in rows if row.get("pass_score", 0.0) >= lower_bound]
    if upper_bound is not None:
        rows = [row for row in rows if row.get("pass_score", 0.0) <= upper_bound]

    rows = sorted(rows, key=lambda x: x.get("pass_score", 0.0), reverse=True)
    return rows


def convert_rejection_samples_to_dataset(
    gcs_path: str, lower_bound: float | None = None, upper_bound: float | None = None
) -> list[Sample]:
    """Convert a list of RejectionSingleSample dicts to a list of Samples."""

    filtered_rejection_rows = filter_rejection_sampled_data(gcs_path, lower_bound, upper_bound)
    return [Sample.from_dict(row) for row in filtered_rejection_rows]
