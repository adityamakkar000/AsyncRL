import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import gcsfs
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.constants import GS_BUCKET, HF_CHECKPOINT_PATH
from src.data import Sample, Verifier, VerifierInput
from src.model import Model
from src.vllm_engine.main import vLLMEngine

from .config import evalConfig
from .utils import fetch_eval_samples

load_dotenv()


class EvalRunner:
    def __init__(self, config: DictConfig | evalConfig):
        self.config = config
        self.vllm_config = config.vllm_config
        self.model_config = config.model_config
        self.verifier = Verifier()

        self.check_config()
        self.vllm_engine = vLLMEngine(self.vllm_config, self.config.max_connections, debug=config.debug)

    def check_config(self):
        if self.model_config.use_best_ckpt and self.model_config.step_number is not None:
            raise ValueError("Cannot set both use_best_ckpt and step_number.")
        if not self.model_config.use_best_ckpt and self.model_config.step_number is None:
            raise ValueError("Must set either use_best_ckpt or step_number.")

    def setup_model(self):
        self.gs_path = f"{GS_BUCKET}/runs/{self.model_config.model_name}"
        config = f"{self.gs_path}/config.json"

        logger.info(f"Loading model config from {config}...")
        fs = gcsfs.GCSFileSystem()
        with fs.open(config.replace("gs://", ""), "r") as f:
            model_config = json.loads(f.read())

        self.train_config = OmegaConf.create(model_config)
        model_config = self.train_config.model_config
        logger.info(f"Model config loaded: \n{OmegaConf.to_yaml(model_config)}")
        logger.info(f"Loading model from {self.gs_path}...")
        model = Model(model_config)
        self.step_number, params = model.load_from_ckpt(
            self.gs_path, step_number=self.model_config.step_number, use_best=self.model_config.use_best_ckpt
        )
        logger.info(f"Checkpoint loaded successfully from step {self.step_number}")
        logger.info("Saving model to HF weights...")

        model.save_hf(HF_CHECKPOINT_PATH, params)
        logger.info("Model saved to HF weights.")

    def launch_vllm(self):
        self.vllm_engine.launch_vllm(HF_CHECKPOINT_PATH)

    def run_task(self, task: str) -> dict[str, float]:
        task_samples: list[Sample] = fetch_eval_samples(task)
        outputs = self.vllm_engine.generate_completions(
            prompts=[sample.prompt for sample in task_samples],
            max_sequence_len=self.config.max_tokens,
            pass_at=self.config.epochs,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        prompt_to_answer = {sample.prompt: sample.answer for sample in task_samples}

        def flatten_2d_list(lst: list[list[Any]]) -> list[float]:
            return [item for sublist in lst for item in sublist]

        rewards = flatten_2d_list(
            [
                [self.verifier(VerifierInput(completion, prompt_to_answer[prompt])) for completion in rollouts]
                for prompt, rollouts in zip(outputs.prompts, outputs.completions)
            ]
        )
        valid_rewards = [r for r in rewards if r is not None]
        avg_reward = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0

        return {"average": avg_reward}

    def process_results(self, results: dict[str, dict[str, float]]):
        logger.info("Evaluation results:")
        for task, metrics in results.items():
            logger.info(f"Task: {task}")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

        # TODO: add visualization and logging to WandB or TextWriter

    def launch_eval(self):
        eval_results = dict()

        with ThreadPoolExecutor(max_workers=self.config.max_connections) as executor:
            futures = {executor.submit(self.run_task, task): task for task in self.config.tasks}
            for future in as_completed(futures):
                task = futures[future]
                eval_results[task] = future.result()

        self.process_results(eval_results)

    def cleanup(self):
        logger.info("Cleaning up vLLM engine...")
        self.vllm_engine.cleanup()

    def run_evaluation(self):
        self.setup_model()
        self.launch_vllm()
        self.launch_eval()
