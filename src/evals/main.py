import json
import os
import subprocess
import time

import gcsfs
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.constants import (
    DISPLAY,
    EVAL_LOG_DIR,
    GPU_MEMORY_UTILIZATION,
    GS_BUCKET,
    HF_CHECKPOINT_PATH,
    IP,
    MAX_TASKS,
    PORT,
    SERVED_MODEL_NAME,
    VLLM_SERVER_TIMEOUT,
)
from src.model import Model

from .config import evalConfig
from .utils import (
    format_command,
    hash_dictConfig,
    ping_server,
    terminate_process,
)

load_dotenv()


class EvalRunner:
    def __init__(self, config: DictConfig | evalConfig):
        self.config = config
        self.vllm_config = config.vllm_config
        self.model_config = config.model_config

        self.check_config()
        self.vllm_process = None

    def check_config(self):
        if self.model_config.use_best_ckpt and self.model_config.step_number is not None:
            raise ValueError("Cannot set both use_best_ckpt and step_number.")
        if not self.model_config.use_best_ckpt and self.model_config.step_number is None:
            raise ValueError("Must set either use_best_ckpt or step_number.")

    def setup_model(self):
        self.gs_path = f"{GS_BUCKET}/{self.model_config.model_name}"
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
        # TODO:
        # after we haev saved our own model
        # use flag vllm serve <path_to_model>
        # https://discuss.vllm.ai/t/how-to-use-local-model-when-using-vllm-serve/1149

        # args: https://docs.vllm.ai/en/v0.5.4/models/engine_args.html
        command = [
            "vllm",
            "serve",
            HF_CHECKPOINT_PATH,
            # variable args
            "--data-parallel-size",
            str(self.vllm_config.data_parallel_size),
            "--tensor-parallel-size",
            str(self.vllm_config.tensor_parallel_size),
            "--max-num-seqs",
            str(self.vllm_config.max_sequences),
            "--max-num-batched-tokens",
            str(self.vllm_config.max_batched_tokens),
            # fixed args
            "--gpu-memory-utilization",
            GPU_MEMORY_UTILIZATION,
            "--port",
            PORT,
            "--disable-log-requests",
            "--enable-prefix-caching",
            "--served-model-name",
            SERVED_MODEL_NAME,
        ]

        logger.info(f"Launching vLLM with command: \n{format_command(command)}")

        # pop off 'cpu' device so VLLM can you tpu
        vllm_env = os.environ.copy()
        vllm_env.pop("JAX_PLATFORMS", None)
        self.vllm_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL if not self.config.debug else None,
            env=vllm_env,
        )
        start = time.perf_counter()
        while not ping_server(IP, PORT):
            logger.info("Waiting for vLLM server to be ready...")
            time.sleep(5)
            if time.perf_counter() - start > VLLM_SERVER_TIMEOUT:
                raise TimeoutError(f"vLLM server did not start within {VLLM_SERVER_TIMEOUT} seconds.")
            if self.vllm_process.poll() is not None:
                raise RuntimeError("vLLM server process has exited unexpectedly.")
        end = time.perf_counter()
        logger.info(f"vLLM server is ready in {end - start:.2f} seconds.")

    def launch_eval(self):
        log_file_template = hash_dictConfig(self.config)
        self.log_path = os.path.join(os.path.abspath(EVAL_LOG_DIR), log_file_template)

        if not os.path.exists(self.log_path):
            logger.info("launching new evaluation ...")
            # args: https://github.com/groq/openbench?tab=readme-ov-file#commands-and-options
            command = (
                [
                    "bench",
                    "eval",
                ]
                + self.config.tasks
                + [
                    # variable args
                    "--max-connections",
                    str(self.config.max_connections),
                    "--epochs",
                    str(self.config.epochs) if not self.config.debug else "1",
                    "--temperature",
                    str(self.config.temperature),
                    "--top-p",
                    str(self.config.top_p),
                    "--max-tokens",
                    str(self.config.max_tokens),
                    # fix args
                    "--model",
                    "vllm/" + SERVED_MODEL_NAME,
                    "--max-tasks",
                    MAX_TASKS,
                    "--log-dir",
                    f"{EVAL_LOG_DIR}/{log_file_template}",
                    "--display",
                    DISPLAY,
                    "--log-samples",
                    "--log-format",
                    "json",
                ]
            )
            if self.config.debug:
                command += ["--limit", "10", "--debug"]

        else:
            """
            #TODO: restarting has some problems with openbench 
             if we really need to restart an eval
             we can just make seperate processes for each task 
             current issues 
             1. eval retry uses the intearactive dashbord dispaly -- doesn't support display arg
             2. it retries but fails with more then one task
            """
            logger.info("resuming evaluation from existing run ...")
            log_files = [f"{self.log_path}/{f}" for f in os.listdir(self.log_path) if f.endswith(".json")]
            command = (
                [
                    "bench",
                    "eval-retry",
                ]
                + log_files
                + [
                    # fixed args
                    "--log-samples",
                ]
            )

        logger.info(f"Launching evaluation with command: \n{format_command(command)}")
        subprocess.run(command)

    def parse_metrics(self):
        eval_path = f"{self.gs_path}/evals"
        fs = gcsfs.GCSFileSystem()
        tasks = [t.replace("_", "-") for t in self.config.tasks]

        def find_task(f_name: str):
            for task in tasks:
                if task in f_name:
                    return task
            raise ValueError(f"Could not find task for file name: {f_name}")

        for f in os.listdir(self.log_path):
            if f.endswith(".json"):
                try:
                    current_task = find_task(f)
                except ValueError:
                    logger.warning(f"Could not find task for file {f}, skipping upload.")
                    continue
                logger.info(f"Uploading metrics for task: {current_task} ...")
                with fs.open(f"{eval_path}/{current_task}", "w") as gcs_f:
                    with open(f"{self.log_path}/{f}", "r") as local_f:
                        gcs_f.write(local_f.read())

        # TODO: maybe wandb but not needed for now

    def cleanup(self):
        logger.info("Cleaning up...")
        self.vllm_process = terminate_process(self.vllm_process, "vLLM")

    def run_evaluation(self):
        self.setup_model()
        self.launch_vllm()
        self.launch_eval()
        self.parse_metrics()
