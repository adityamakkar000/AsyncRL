import asyncio

import hydra
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.data.config import Sample
from src.data.register import GLOBAL_DICT
from src.data.utils import upload_local_file_to_gcs
from src.data.verifier import Verifier, VerifierInput
from src.rejection_sampling.config import RejectionSingleSample, rejectionSamplingConfig
from src.vllm_engine.main import vLLMEngine, vLLMOutput


class RejectionSample:
    def __init__(self, config):
        self.config = config
        self.verifier = Verifier()

        self.check_config()
        self.vllm_engine = vLLMEngine(config.vllm_config, self.config.max_workers)

        self.num_samples = config.num_samples

    def check_config(self):
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

        logger.info(f"Uploading to gcs {gcs_path}...")

        upload_local_file_to_gcs(local_path, gcs_path)

    def get_dataset(self, dataset_name: str) -> list[Sample]:
        return GLOBAL_DICT[dataset_name]()[: self.num_samples]

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

    async def get_samples(self, samples: list[Sample]) -> list[RejectionSingleSample]:
        """Generates num_samples completions for the given sample and returns a RejectionSingleSample with the pass score."""
        prompts = [sample.prompt for sample in samples]
        vllm_output: vLLMOutput = await self.vllm_engine.generate_completions(
            prompts, self.config.max_sequence_len, self.config.pass_at, self.config.temperature
        )

        output_samples = []

        for sample, completions in zip(samples, vllm_output.completions):
            output_samples.append(self.convert_to_rejection_sample(sample, completions))

        return output_samples

    def cleanup(self):
        logger.info("Clearning up vLLM engine...")
        self.vllm_engine.cleanup()

    async def run(self):
        self.vllm_engine.launch_vllm(self.config.hf_model_name)
        for dataset_gcs, gcs_path in zip(self.config.datasets, self.config.gcs_paths):
            samples = self.get_dataset(dataset_gcs)
            logger.info(f"Running rejection sampling on {len(samples)} samples from {dataset_gcs}...")

            rejection_samples = await self.get_samples(samples)
            self.upload_dataset(rejection_samples, dataset_gcs, gcs_path)

        self.cleanup()


cs = ConfigStore.instance()
cs.store(name="base", node=rejectionSamplingConfig)


@hydra.main(version_base=None, config_path="./configs/rejection_sample_config", config_name="main")
def main(cfg: DictConfig) -> None:
    logger.info(f"Rejection Sampling Configuration: \n{OmegaConf.to_yaml(cfg)}")

    rejection_sample = RejectionSample(config=cfg)
    try:
        asyncio.run(rejection_sample.run())
    finally:
        rejection_sample.cleanup()


if __name__ == "__main__":
    main()
