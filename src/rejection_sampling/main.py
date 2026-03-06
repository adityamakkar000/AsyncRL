from loguru import logger

from src.data import load_jsonl_from_gcs
from src.data.config import RejectionSingleSample, Sample
from src.data.register import GLOBAL_DICT
from src.data.verifier import Verifier, VerifierInput
from src.vllm_engine.main import vLLMEngine


class RejectionSample:
    def __init__(self, config):
        self.config = config
        self.vllm_engine = vLLMEngine(config.vllm_config)
        self.verifier = Verifier()

        self.check_config()
        self.num_samples = config.num_samples

    def check_config(self):
        for dataset in self.config.datasets:
            if dataset not in GLOBAL_DICT.keys():
                raise ValueError(f"Dataset {dataset} is not supported for Rejection Sampling.")

    def upload_dataset(self, rejection_samples: list[Sample]):
        pass

    def get_dataset(self, dataset_name: str) -> list[Sample]:
        rows = load_jsonl_from_gcs(dataset_name)
        if not rows:
            raise ValueError(f"No rows found at {dataset_name}")
        samples = [Sample.from_dict(r) for r in rows]
        return samples

    def get_reward(self, prompt: str, completion: str, reference_answer: str) -> float | None:
        """Calls the verifier to get the reward for a given prompt and completion."""
        return self.verifier(VerifierInput(completion, reference_answer))

    def generate_completions(self, prompt: str) -> list[str]:
        """Calls the vLLM server to generate completions for a given prompt."""

        pass

    def pass_at_k(self, sample: Sample) -> RejectionSingleSample:
        """Generates num_samples completions for the given sample and returns a RejectionSingleSample with the pass score."""
        completions = self.generate_completions(sample.prompt)
        rewards = [self.get_reward(sample.prompt, c, sample.answer) for c in completions]
        pass_score = sum(r for r in rewards) / len(rewards)
        return RejectionSingleSample(
            prompt=sample.prompt, answer=sample.answer, solution=sample.solution, pass_score=pass_score
        )

    def get_samples(self, samples: list[Sample]) -> list[RejectionSingleSample]:
        rejection_samples = [self.pass_at_k(sample) for sample in samples]
        return rejection_samples

    def cleanup(self):
        logger.info("Clearning up vLLM engine...")
        self.vllm_engine.cleanup()

    def run(self):
        self.vllm_engine.launch_vllm(self.config.model_name)
        for dataset in self.config.datasets:
            samples = self.get_dataset(dataset)[: self.num_samples]
            logger.info(f"Running rejection sampling on {len(samples)} samples from {dataset}...")

            rejection_samples = self.get_samples(samples)
            self.upload_dataset(rejection_samples)
