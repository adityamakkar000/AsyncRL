import os
import random

from omegaconf import DictConfig

from src.constants import DATA, GS_BUCKET
from .config import Sample
from .register import GLOBAL_DICT
from .utils import delete_local_file, samples_to_dataset, upload_local_file_to_gcs, write_dataset_to_local_jsonl

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
END = "\033[0m"


class ProcessDataset:
    """Process a dataset (registered or HuggingFace) and upload to GCS."""

    def __init__(self, cfg: DictConfig) -> None:
        self.datasets_name = cfg.name
        self.chunk_size = cfg.chunk_size
        self._resolve_config()

        random.seed(cfg.seed)

    def _resolve_config(self) -> None:
        """Resolve: registered dataset (name in GLOBAL_DICT)"""
        for name in self.datasets_name:
            if name not in GLOBAL_DICT:
                raise ValueError(
                    f"Unknown dataset: {name}. "
                    f"Use a registered name (in src/data/register.py), set hf_path+columns, or add to datasets. "
                    f"Registered: {list(GLOBAL_DICT.keys())}"
                )

    def get_samples(self, name) -> list[Sample]:
        return GLOBAL_DICT[name]()

    @staticmethod
    def apply_transformations(samples: list[Sample]) -> list[Sample]:
        random.shuffle(samples)
        return samples

    def _process_and_upload(self, name: str) -> None:
        """Load dataset (from registry or HuggingFace), process, and upload chunks to GCS."""
        samples = ProcessDataset.apply_transformations(self.get_samples(name))
        dataset = samples_to_dataset(samples)

        base_gs = f"{GS_BUCKET}/{DATA}/{name}"
        cwd = os.getcwd()
        base_name = f"{name}.jsonl"

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

    def process_and_upload(self):
        for name in self.datasets_name:
            print(f"{YELLOW}Processing dataset: {name}{END}")
            self._process_and_upload(name)
