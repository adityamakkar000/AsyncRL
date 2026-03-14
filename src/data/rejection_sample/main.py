import json
import os

from loguru import logger

from src.data.config import Sample
from src.data.register import GLOBAL_DICT
from src.data.rejection_sample.config import RejectionSingleSample
from src.data.utils import load_jsonl_from_gcs, upload_local_file_to_gcs
from src.data.verifier import Verifier, VerifierInput
from src.vllm_engine.main import vLLMEngine, vLLMOutput


class RejectionSample:
    def __init__(self, config):
        self.config = config
        self.verifier = Verifier()

        self.check_config()
        self.vllm_engine = vLLMEngine(config.vllm_config, self.config.max_workers)

        self.num_samples = config.num_samples
        self.checkpoint_number = 32

    def check_config(self):
        assert len(self.config.datasets) == len(self.config.gcs_paths), (
            "Number of datasets and gcs_paths must be the same."
        )

    def get_existing_samples(self, gcs_path: str) -> list[dict]:
        """Get existing samples from GCS in a json format."""
        return load_jsonl_from_gcs(gcs_path)

    def upload_dataset(self, rejection_samples: list[RejectionSingleSample], dataset_name: str, gcs_path: str):
        """Uploads the rejection samples to GCS."""
        samples_dict = [s.get_dict() for s in rejection_samples]
        existing_samples = self.get_existing_samples(gcs_path)

        total_samples = len(existing_samples) + len(samples_dict)
        local_path = f"{dataset_name}_{total_samples}_rejection_samples.jsonl"

        if len(existing_samples) > 0:
            samples_dict = existing_samples + samples_dict

        with open(local_path, "w") as f:
            for s in samples_dict:
                f.write(json.dumps(s) + "\n")

        logger.info(f"Uploading to gcs {gcs_path}...")

        upload_local_file_to_gcs(local_path, gcs_path)

        try:
            os.remove(local_path)
        except OSError:
            pass

    def get_dataset(self, dataset_name: str) -> list[Sample]:
        dataset = GLOBAL_DICT[dataset_name]()
        if self.num_samples == -1:
            return dataset

        logger.warning(f"Using only {self.num_samples}")
        return dataset[: self.num_samples]

    def filter_samples(self, samples: list[Sample], gcs_path: str):
        existing_rows = self.get_existing_samples(gcs_path)
        if len(existing_rows) == 0:
            return samples

        existing_prompts = set(row["prompt"] for row in existing_rows)
        filtered_samples = [s for s in samples if s.prompt not in existing_prompts]
        return filtered_samples

    def get_reward(self, prompt: str, completion: str, reference_answer: str) -> float | None:
        """Calls the verifier to get the reward for a given prompt and completion."""
        return self.verifier(VerifierInput(completion, reference_answer))

    def convert_to_rejection_sample(self, sample: Sample, completions: list[str]) -> RejectionSingleSample:
        rewards = [self.get_reward(sample.prompt, c, sample.answer) for c in completions]
        valid_rewards = [r for r in rewards if r is not None]
        pass_score = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0

        return RejectionSingleSample(
            prompt=sample.prompt, answer=sample.answer, solution=sample.solution, pass_score=pass_score
        )

    def generate_output_samples(self, samples: list[Sample], vllm_output: vLLMOutput, gcs_path: str, dataset_name: str):
        output_samples = []
        for sample, completions in zip(samples, vllm_output.completions):
            output_sample = self.convert_to_rejection_sample(sample, completions)
            output_samples.append(output_sample)
            if len(output_samples) % self.checkpoint_number == 0:
                self.upload_dataset(output_samples, dataset_name, gcs_path)
                output_samples = []
        
        self.upload_dataset(output_samples, dataset_name, gcs_path)

    def rejection_sample(self, samples: list[Sample], gcs_path: str, name: str):
        """Generates num_samples completions for the given sample and returns a RejectionSingleSample with the pass score."""
        prompts = [sample.prompt for sample in samples]
        vllm_output: vLLMOutput = self.vllm_engine.generate_completions(
            prompts, self.config.max_sequence_len, self.config.pass_at, self.config.temperature
        )

        self.generate_output_samples(samples, vllm_output, gcs_path, name)

    def cleanup(self):
        logger.info("Clearning up vLLM engine...")
        self.vllm_engine.cleanup()

    def run(self):
        self.vllm_engine.launch_vllm(self.config.hf_model_name)
        for dataset, gcs_path in zip(self.config.datasets, self.config.gcs_paths):
            samples = self.filter_samples(self.get_dataset(dataset), gcs_path)

            if len(samples) == 0:
                logger.info(f"No new samples to process for {dataset}. Skipping...")
                continue

            logger.info(f"Running rejection sampling on {len(samples)} samples from {dataset}...")

            self.rejection_sample(samples, gcs_path, dataset)

        self.cleanup()
