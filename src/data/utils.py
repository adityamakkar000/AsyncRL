import gcsfs
from datasets import Dataset
import os

def upload_local_file_to_gcs(local_path: str, gs_path: str):
    """Upload a local file to GCS. gs_path should be like 'gs://bucket/path/file.jsonl'."""
    fs = gcsfs.GCSFileSystem()
    gcs_path = gs_path.replace("gs://", "") if gs_path.startswith("gs://") else gs_path

    with open(local_path, "rb") as src:
        data = src.read()
        with fs.open(gcs_path, "wb") as dst:
            dst.write(data) # pyright: ignore[reportArgumentType]
def write_dataset_to_local_jsonl(dataset: Dataset, local_path: str) -> None:
    """Write dataset to a local JSONL file (all columns)."""

    dataset.to_json(local_path, lines=True)

def delete_local_file(local_path: str) -> None:
    """Remove the local file to free disk. Make sure to only remove .jsonl files"""
    if os.path.isfile(local_path) and local_path.endswith(".jsonl"):
        os.remove(local_path)