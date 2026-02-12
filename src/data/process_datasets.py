import argparse
import os
from typing import List
from datasets import Dataset, load_dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import COLUMNS, DATA, DATASETS, GS_BUCKET, HF_PATHS
from utils import write_dataset_to_local_jsonl, upload_local_file_to_gcs, delete_local_file

CHUNK_SIZE = 50_000 # max number of rows per jsonl file for larger datasets
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
END = "\033[0m"

def _rename_messages_to_trace_if_present(dataset: Dataset) -> Dataset:
    """If the dataset has a 'messages' column, rename it to 'trace'. Otherwise return as-is."""
    if "messages" in dataset.column_names:
        print(f"{GREEN}Renaming 'messages' column to 'trace'.{END}")
        return dataset.rename_column("messages", "trace")
    return dataset

def process_and_upload_dataset(
    dataset_name: str,
    hf_path: str,
    columns: List[str],
    split: str = "train",
) -> None:
    
    print(f"{YELLOW}Loading dataset from HuggingFace: {hf_path} (split={split})...{END}")
    dataset = load_dataset(hf_path, split=split) # get the dataset from HF
    print(f"{GREEN}Loaded dataset with {len(dataset)} rows and columns: {dataset.column_names}{END}")

    # remove columns that are not in the columns list
    if columns is not None:
        original_columns = dataset.column_names
        columns_to_remove = [c for c in original_columns if c not in columns]
        if columns_to_remove:
            print(f"{YELLOW}Removing unused columns: {columns_to_remove}{END}")
        dataset = dataset.remove_columns(columns_to_remove)

    # optional: rename "messages" to "trace" for clarity
    dataset = _rename_messages_to_trace_if_present(dataset)

    base_gs = f"{GS_BUCKET}/{DATA}/{dataset_name}"
    cwd = os.getcwd()
    base_name = f"{dataset_name}.jsonl"

    n = len(dataset)
    print(f"{YELLOW}Preparing to write and upload dataset ({n} rows) to GCS at: {base_gs}{END}")

    iteration = 0
    start_idx = 0

    while start_idx < n:
        end_idx = min(start_idx + CHUNK_SIZE, n)
        print(f"{YELLOW}Processing chunk {iteration}: rows {start_idx} to {end_idx-1}{END}")
        chunk = dataset.select(range(start_idx, end_idx))
        chunk_name = f"{iteration:03d}_{base_name}"
        local_path = os.path.join(cwd, chunk_name)

        print(f"{YELLOW}Writing chunk to {local_path}{END}")
        write_dataset_to_local_jsonl(chunk, local_path)
        print(f"{YELLOW}Uploading chunk to GCS at {base_gs}/{chunk_name}{END}")
        upload_local_file_to_gcs(local_path, f"{base_gs}/{chunk_name}")
        delete_local_file(local_path)
        print(f"{GREEN}Finished chunk {iteration}. Local chunk file deleted.\n{END}")
        
        iteration += 1
        start_idx = end_idx

    print(f"{GREEN}All chunks processed and uploaded.{END}")
    print(f"{RED}Cleaning up cache files.{END}")
    dataset.cleanup_cache_files()
    print(f"{GREEN}Cache cleaned up.{END}")

def main():
    parser = argparse.ArgumentParser(
        description="Process a HuggingFace dataset and upload to GCS (one-time per dataset)."
    )
    hf_group = parser.add_mutually_exclusive_group()
    hf_group.add_argument(
        "--dataset-name",
        type=str,
        choices=DATASETS,
        help="One of the DATASETS from src.constants.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (default: train).",
    )
    hf_group.add_argument(
        "--hf",
        type=str,
        help="optional, provide dataset path for a new dataset (if specified, --name and --columns is required)",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="dataset name for internal use if using --hf option (required if --hf is specified)",
    )
    parser.add_argument(
        "--columns",
        type=str,
        help="the columns you want to keep, seperated by \"/\". Assumes they exist in the dataset (or will be skipped) - required if --hf is specified"
    )
    args = parser.parse_args()

    if args.hf:
        # we are given a new hf path
        if not args.name or not args.columns:
            raise ValueError("name or columsn not passed in for the dataset, make sure they're correct")
        hf_path = args.hf
        dataset_name = args.name
        columns = args.columns.split("/")

        print(f"Starting NEW dataset processing for: {dataset_name} found at {hf_path}")
        process_and_upload_dataset(dataset_name, hf_path, columns, split=args.split)
        return

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

    print(f"Using HuggingFace path: {hf_path}")
    print(f"Columns to keep: {columns}")

    process_and_upload_dataset(args.dataset_name, hf_path, columns, split=args.split)

if __name__ == "__main__":
    main()
