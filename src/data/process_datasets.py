import os
import random

from omegaconf import DictConfig

from src.constants import DATA, GS_BUCKET
from src.data.config import Sample
from src.data.register import GLOBAL_DICT
from src.data.utils import delete_local_file, samples_to_dataset, upload_local_file_to_gcs, write_dataset_to_local_jsonl

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
END = "\033[0m"


class ProcessDataset:
    """Process a dataset (registered or HuggingFace) and upload to GCS."""

    def __init__(self, cfg: DictConfig) -> None:
        self.dataset_name = cfg.name
        self.seed = cfg.seed
        self.chunk_size = cfg.chunk_size
        self._resolve_config()

    def _resolve_config(self) -> None:
        """Resolve: registered dataset (name in GLOBAL_DICT)"""
        name = self.dataset_name
        if name not in GLOBAL_DICT:
            raise ValueError(
                f"Unknown dataset: {name}. "
                f"Use a registered name (in src/data/register.py), set hf_path+columns, or add to datasets. "
                f"Registered: {list(GLOBAL_DICT.keys())}"
            )

    @property
    def get_samples(self) -> list[Sample]:
        return GLOBAL_DICT[self.dataset_name]()

    @staticmethod
    def apply_transformations(samples: list[Sample], seed: int) -> list[Sample]:
        random.seed(seed)
        random.shuffle(samples)
        return samples

    def _process_and_upload(self) -> None:
        """Load dataset (from registry or HuggingFace), process, and upload chunks to GCS."""
        samples = ProcessDataset.apply_transformations(self.get_samples, seed=self.seed)
        dataset = samples_to_dataset(samples)

        base_gs = f"{GS_BUCKET}/{DATA}/{self.dataset_name}"
        cwd = os.getcwd()
        base_name = f"{self.dataset_name}.jsonl"

        n = len(dataset)
        print(f"{YELLOW}Preparing to write and upload samples ({n} rows) to GCS at: {base_gs}{END}")

        iteration = 0
        start_idx = 0

        while start_idx < n:
            end_idx = min(start_idx + self.chunk_size, n)
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