import json
import os

import gcsfs
import lm_eval
from dotenv import load_dotenv
from lm_eval.loggers import WandbLogger
from lm_eval.tasks import TaskManager
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.constants import GS_BUCKET, HF_CHECKPOINT_PATH, SERVED_MODEL_NAME
from src.model import Model
from src.vllm_engine.main import vLLMEngine

from .config import evalConfig
from .eval_model import LocalModelEval

load_dotenv()


class EvalRunner:
    def __init__(self, config: DictConfig | evalConfig):
        self.config = config
        self.vllm_config = config.vllm_config
        self.model_config = config.model_config

        self.vllm_engine = vLLMEngine(self.vllm_config, max_workers=100, debug=config.debug)

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
        self.step_number, params, metadata = model.load_from_ckpt_old(
            self.gs_path, step_number=self.model_config.step_number
        )
        logger.info(f"Checkpoint loaded successfully from step {self.step_number}")
        logger.info("Saving model to HF weights...")

        model.save_hf(HF_CHECKPOINT_PATH, params)
        logger.info("Model saved to HF weights.")
        logger.info(f"Model setup complete with step number {self.step_number} and metadata {metadata}.")

    def launch_vllm(self):
        self.vllm_engine.launch_vllm(HF_CHECKPOINT_PATH)

    def run_evals(self):
        tags = [self.model_config.model_name, f"step_{self.step_number}"] + [t for t in self.config.tasks]
        for i in range(len(tags)):
            tags[i] = tags[i][:64]  # max length is 64 characters

        wandb_logger = WandbLogger(
            init_args={
                "project": "eval_debug",
                "name": f"eval-{self.model_config.model_name}-step-{self.step_number}",
                "tags": tags,
                "entity": os.getenv("WANDB_ENTITY", ""),
            },
            config_args=OmegaConf.to_object(self.config),
        )

        model_args = {
            "model": SERVED_MODEL_NAME,
            "tokenizer": HF_CHECKPOINT_PATH,
            "num_concurrent": self.config.max_concurrent_requests,
        }

        gen_kwargs = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "do_sample": True,
        }

        task_manager = TaskManager(include_path="./src/configs/eval/tasks_yaml/")
        tasks = list(task_manager.load(list(self.config.tasks))["tasks"].values())
        for task in tasks:
            task.set_config(key="repeats", value=self.config.epochs)

        results = lm_eval.simple_evaluate(
            model=LocalModelEval(**model_args),
            apply_chat_template=True,
            num_fewshot=0,
            gen_kwargs=gen_kwargs,
            tasks=tasks,
            task_manager=task_manager,
            log_samples=True,
        )
        logger.info(f"{lm_eval.utils.make_table(results)}")

        wandb_logger.post_init(results)
        wandb_logger.log_eval_result()
        if results.get("samples"):
            wandb_logger.log_eval_samples(results["samples"])
        wandb_logger.run.finish()

    def cleanup(self):
        self.vllm_engine.cleanup()

    def run_evaluation(self):
        self.setup_model()
        self.launch_vllm()
        try:
            self.run_evals()
        finally:
            self.cleanup()
