import asyncio
from threading import Semaphore

from loguru import logger

from src.data.config import Sample
from src.data.register import GLOBAL_DICT
from src.data.utils import upload_local_file_to_gcs
from src.data.verifier import Verifier, VerifierInput
from src.rejection_sampling.config import RejectionSingleSample
from src.vllm_engine.main import vLLMEngine

MAX_CONNECTIONS = 200


class RejectionSample:
    def __init__(self, config):
        self.config = config
        self.vllm_engine = vLLMEngine(config.vllm_config)
        self.verifier = Verifier()
        self.rejection_semaphore = Semaphore(MAX_CONNECTIONS)

        self.check_config()
        self.num_samples = config.num_samples

    def check_config(self):
        for dataset in self.config.datasets:
            if dataset not in GLOBAL_DICT.keys():
                raise ValueError(f"Dataset {dataset} is not supported for Rejection Sampling.")

        assert len(self.config.datasets) == len(self.config.gcs_paths), (
            "Number of datasets and gcs_paths must be the same."
        )

    def upload_dataset(self, rejection_samples: list[RejectionSingleSample], dataset_name: str, gcs_path: str):
        """Uploads the rejection samples to GCS."""
        samples_dict = [s.get_dict() for s in rejection_samples]

        local_path = f"{dataset_name}_{len(samples_dict)}_rejection_samples.jsonl"
        with open(local_path, "w") as f:
            for s in samples_dict:
                f.write(f"{s}\n")

        upload_local_file_to_gcs(local_path, gcs_path)

    def get_dataset(self, dataset_name: str) -> list[Sample]:
        return GLOBAL_DICT[dataset_name]()[: self.num_samples]

    def get_reward(self, prompt: str, completion: str, reference_answer: str) -> float | None:
        """Calls the verifier to get the reward for a given prompt and completion."""
        return self.verifier(VerifierInput(completion, reference_answer))

    async def generate_completions(self, prompt: str) -> list[str]:
        """Calls the vLLM server to generate completions for a given prompt."""
        with self.rejection_semaphore:
            response = await self.vllm_engine.client.completions.create(
                model=self.config.model_name,
                prompt=prompt,
                max_tokens=self.config.vllm_config.max_tokens,
                temperature=self.config.temperature,
                n=self.config.pass_at,
            )
            return [choice.text for choice in response.choices]

    async def pass_at_k(self, sample: Sample) -> RejectionSingleSample:
        """Generates num_samples completions for the given sample and returns a RejectionSingleSample with the pass score."""
        completions = await self.generate_completions(sample.prompt)

        rewards = [self.get_reward(sample.prompt, c, sample.answer) for c in completions]
        valid_rewards = [r for r in rewards if r is not None]
        pass_score = sum(r for r in valid_rewards) / len(valid_rewards) if valid_rewards else 0
        return RejectionSingleSample(
            prompt=sample.prompt, answer=sample.answer, solution=sample.solution, pass_score=pass_score
        )

    async def get_samples(self, samples: list[Sample]) -> list[RejectionSingleSample]:
        rejection_samples = await asyncio.gather(*[self.pass_at_k(sample) for sample in samples])
        return rejection_samples

    def cleanup(self):
        logger.info("Clearning up vLLM engine...")
        self.vllm_engine.cleanup()

    async def run(self):
        self.vllm_engine.launch_vllm(self.config.model_name)
        for dataset, gcs_path in zip(self.config.datasets, self.config.gcs_paths):
            samples = self.get_dataset(dataset)
            logger.info(f"Running rejection sampling on {len(samples)} samples from {dataset}...")

            rejection_samples = await self.get_samples(samples)
            self.upload_dataset(rejection_samples, dataset, gcs_path)

        self.cleanup()
