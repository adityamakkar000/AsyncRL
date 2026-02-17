import os

import hydra
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf

from src.constants import DATA, GS_BUCKET
from src.data.utils import delete_local_file, upload_local_file_to_gcs, write_dataset_to_local_jsonl

CHUNK_SIZE = 50_000  # max number of rows per jsonl file for larger datasets
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


class ProcessDataset:
    """Process a HuggingFace dataset and upload to GCS."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._resolve_config()

    def _resolve_config(self) -> None:
        """Resolve dataset_name vs custom hf_path/name/columns."""
        if self.cfg.hf_path is not None:
            if not self.cfg.name or not self.cfg.columns:
                raise ValueError("name and columns are required when hf_path is specified")
            self.dataset_name = self.cfg.name
            self.hf_path = self.cfg.hf_path
            self.columns = list(self.cfg.columns) if isinstance(self.cfg.columns, (list, tuple)) else [self.cfg.columns]
        else:
            if self.cfg.dataset_name is None:
                raise ValueError("Either dataset_name or hf_path must be set")
            if self.cfg.dataset_name not in self.cfg.datasets:
                raise ValueError(
                    f"Unknown dataset_name: {self.cfg.dataset_name}. Available: {list(self.cfg.datasets.keys())}"
                )
            preset = self.cfg.datasets[self.cfg.dataset_name]
            self.dataset_name = self.cfg.dataset_name
            self.hf_path = preset.hf_path
            self.columns = list(preset.columns)
        self.split = self.cfg.split

    def _process_and_upload(self) -> None:
        """Load dataset from HuggingFace, process, and upload chunks to GCS."""
        print(f"{YELLOW}Loading dataset from HuggingFace: {self.hf_path} (split={self.split})...{END}")
        dataset = load_dataset(self.hf_path, split=self.split)
        print(f"{GREEN}Loaded dataset with {len(dataset)} rows and columns: {dataset.column_names}{END}")

        if self.columns is not None:
            original_columns = dataset.column_names
            columns_to_remove = [c for c in original_columns if c not in self.columns]
            if columns_to_remove:
                print(f"{YELLOW}Removing unused columns: {columns_to_remove}{END}")
            dataset = dataset.remove_columns(columns_to_remove)

        dataset = _rename_messages_to_trace_if_present(dataset)

        base_gs = f"{GS_BUCKET}/{DATA}/{self.dataset_name}"
        cwd = os.getcwd()
        base_name = f"{self.dataset_name}.jsonl"

        n = len(dataset)
        print(f"{YELLOW}Preparing to write and upload dataset ({n} rows) to GCS at: {base_gs}{END}")

        iteration = 0
        start_idx = 0

        while start_idx < n:
            end_idx = min(start_idx + CHUNK_SIZE, n)
            print(f"{YELLOW}Processing chunk {iteration}: rows {start_idx} to {end_idx - 1}{END}")
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

    def run(self) -> None:
        """Run the dataset processing pipeline."""
        self._process_and_upload()


@hydra.main(version_base=None, config_path="../configs/process_datasets")
def main(cfg: DictConfig) -> None:
    """Entry point for Hydra."""
    print("ProcessDataset configuration:\n" + OmegaConf.to_yaml(cfg))
    processor = ProcessDataset(cfg)
    processor.run()


if __name__ == "__main__":
    main()
