import json
import os

import gcsfs
import lm_eval
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from stax import TextWriter, WandBWriter

from src.constants import GS_BUCKET, HF_CHECKPOINT_PATH, IP, PORT, SERVED_MODEL_NAME
from src.model import Model
from src.vllm_engine.main import vLLMEngine

from .config import evalConfig

load_dotenv()


class EvalRunner:
    def __init__(self, config: DictConfig | evalConfig):
        self.config = config
        self.vllm_config = config.vllm_config
        self.model_config = config.model_config

        self.check_config()
        self.vllm_engine = vLLMEngine(self.vllm_config, max_workers=100, debug=config.debug)

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
        self.step_number, params, metadata = model.load_from_ckpt(
            self.gs_path, step_number=self.model_config.step_number, use_best=self.model_config.use_best_ckpt
        )
        logger.info(f"Checkpoint loaded successfully from step {self.step_number}")
        logger.info("Saving model to HF weights...")

        model.save_hf(HF_CHECKPOINT_PATH, params)
        logger.info("Model saved to HF weights.")

        if writer_config := self.train_config.wandb_config:
            if metadata["writer_id"] is None:
                raise ValueError("Writer ID is None in metadata, cannot initialize WandBWriter.")
            self.writer = WandBWriter(
                entity=os.getenv("WANDB_ENTITY", ""),
                project=writer_config.project,
                metrics_to_print=dict(),
                run_id=metadata["writer_id"],
            )

        else:
            self.writer = TextWriter(metrics_to_print=dict())

        logger.info(f"Model setup complete with step number {self.step_number} and metadata {metadata}.")

    def launch_vllm(self):
        self.vllm_engine.launch_vllm(HF_CHECKPOINT_PATH)

    def run_lm_eval_task(self, tasks: list[str]):
        model_args = (
            f"model={SERVED_MODEL_NAME},base_url=http://{IP}:{PORT}/v1/completions,tokenizer={HF_CHECKPOINT_PATH}"
        )

        results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args=model_args,
            tasks=tasks,
            num_fewshot=0,
            batch_size=100,
            repeats=self.config.epochs,
        )
        logger.info(f"{lm_eval.utils.make_table(results)}")

    def process_results(self, results: dict[str, dict[str, float]]):
        logger.info("Evaluation results:")

        self.writer.log_eval_results(
            self.step_number,
            results,
        )

        for task, metrics in results.items():
            logger.info(f"Task:\t{task}")
            for metric, value in metrics.items():
                logger.info(f"\t\t{metric}: {value:.4f}")

    def launch_eval(self):
        eval_results = self.run_lm_eval_task(self.config.tasks)
        # self.process_results(eval_results)

    def cleanup(self):
        self.vllm_engine.cleanup()
        self.writer.finish()

    def run_evaluation(self):
        self.setup_model()
        self.launch_vllm()
        try:
            self.launch_eval()
        finally:
            self.cleanup()
