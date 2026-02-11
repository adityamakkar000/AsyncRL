import argparse
import json
import os
from typing import List
from datasets import Dataset, load_dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import COLUMNS, DATA, DATASETS, GS_BUCKET, HF_PATHS
from utils import write_dataset_to_local_jsonl, upload_local_file_to_gcs, delete_local_file

CHUNK_SIZE = 50_000 # max number of rows per jsonl file for larger datasets

def _rename_messages_to_trace_if_present(dataset: Dataset) -> Dataset:
    """If the dataset has a 'messages' column, rename it to 'trace'. Otherwise return as-is."""
    if "messages" in dataset.column_names:
        print("Renaming 'messages' column to 'trace'.")
        return dataset.rename_column("messages", "trace")
    return dataset

def process_and_upload_dataset(
    dataset_name: str,
    hf_path: str,
    columns: List[str],
    split: str = "train",
) -> None:
    
    print(f"Loading dataset from HuggingFace: {hf_path} (split={split})...")
    dataset = load_dataset(hf_path, split=split) # get the dataset from HF
    print(f"Loaded dataset with {len(dataset)} rows and columns: {dataset.column_names}")

    # remove columns that are not in the columns list
    if columns is not None:
        original_columns = dataset.column_names
        columns_to_remove = [c for c in original_columns if c not in columns]
        if columns_to_remove:
            print(f"Removing unused columns: {columns_to_remove}")
        dataset = dataset.remove_columns(columns_to_remove)

    # optional: rename "messages" to "trace" for clarity
    dataset = _rename_messages_to_trace_if_present(dataset)

    base_gs = f"{GS_BUCKET}/{DATA}/{dataset_name}"
    cwd = os.getcwd()
    base_name = f"{dataset_name}.jsonl"

    n = len(dataset)
    print(f"Preparing to write and upload dataset ({n} rows) to GCS at: {base_gs}")

    # if the dataset is small enough, write to a single file
    if n <= CHUNK_SIZE:
        local_path = os.path.join(cwd, base_name)
        print(f"Writing entire dataset to single JSONL file: {local_path}")
        write_dataset_to_local_jsonl(dataset, local_path)
        print(f"Uploading {local_path} to GCS at {base_gs}.jsonl ...")
        upload_local_file_to_gcs(local_path, f"{base_gs}.jsonl")
        delete_local_file(local_path)
        print("Upload complete. Local file deleted.")
        return

    # if the dataset is larger than the chunk size, split into chunks and upload each chunk to GCS
    print(f"Dataset is large, splitting into chunks of size {CHUNK_SIZE}...")

    iteration = 0
    start_idx = 0

    while start_idx < n:
        end_idx = min(start_idx + CHUNK_SIZE, n)
        print(f"Processing chunk {iteration}: rows {start_idx} to {end_idx-1}")
        chunk = dataset.select(range(start_idx, end_idx))
        chunk_name = f"{iteration}_{base_name}"
        local_path = os.path.join(cwd, chunk_name)

        print(f"Writing chunk to {local_path}")
        write_dataset_to_local_jsonl(chunk, local_path)
        print(f"Uploading chunk to GCS at {base_gs}/{chunk_name}")
        upload_local_file_to_gcs(local_path, f"{base_gs}/{chunk_name}")
        delete_local_file(local_path)
        print(f"Finished chunk {iteration}. Local chunk file deleted.\n")
        
        iteration += 1
        start_idx = end_idx

    print("All chunks processed and uploaded.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a HuggingFace dataset and upload to GCS (one-time per dataset)."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=DATASETS,
        help="One of the DATASETS from src.constants.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (default: train).",
    )
    args = parser.parse_args()

    print(f"Starting dataset processing for: {args.dataset_name} (split={args.split})")

    if args.dataset_name not in DATASETS:
        raise ValueError(f"Dataset {args.dataset_name} not in supported DATASETS: {DATASETS}")

    idx = DATASETS.index(args.dataset_name)
    if idx >= len(HF_PATHS) or idx >= len(COLUMNS):
        raise ValueError(
            f"No HF_PATHS/COLUMNS entry for dataset index {idx}. "
            "Check src.constants (DATASETS, HF_PATHS, COLUMNS lengths)."
        )
    hf_path = HF_PATHS[idx]
    columns = COLUMNS[idx]

    if args.dataset_name == "omnimath": # omni math custom logic
        print("Detected omnimath: using split 'test' for processing.")
        args.split = "test"

    print(f"Using HuggingFace path: {hf_path}")
    print(f"Columns to keep: {columns}")

    process_and_upload_dataset(args.dataset_name, hf_path, columns, split=args.split)


if __name__ == "__main__":
    main()
