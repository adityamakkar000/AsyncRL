import os
import subprocess
import time

from loguru import logger
from omegaconf import DictConfig

from src.evals.utils import (
    format_command,
    hash_dictConfig,
    ping_server,
    terminate_process,
)

SERVED_MODEL_NAME = "eval_model"
IP = "localhost"
PORT = "8000"
GPU_MEMORY_UTILIZATION = "0.98"
EVAL_LOG_DIR = "logs/eval_logs"
DISPLAY = "plain"
MAX_TASKS = "5"


class EvalRunner:
    def __init__(self, config: DictConfig):
        self.config = config
        self.vllm_config = config.vllm_config
        self.model_config = config.model_config

        self.check_config()
        self.vllm_process = None

    def check_config(self):
        # model config
        if self.model_config.use_best_ckpt and self.model_config.step_number is not None:
            raise ValueError("Cannot set both use_best_ckpt and step_number.")
        if not self.model_config.use_best_ckpt and self.model_config.step_number is None:
            raise ValueError("Must set either use_best_ckpt or step_number.")

        # vllm config
        if isinstance(self.config.vllm_config.max_batched_tokens, str):
            if self.config.vllm_config.max_batched_tokens != "auto":
                raise ValueError("If max_batched_tokens is a string, it must be 'auto'.")
        dtypes = ["auto", "bfloat16", "float16", "float32"]
        if self.config.vllm_config.dtype not in dtypes:
            raise ValueError(f"dtype must be one of {dtypes}, but got {self.config.vllm_config.dtype}.")

    def load_model(self):
        """
        #TODO:
        load the model from checkpoint
        setup wandb

        """

        raise NotImplementedError()

    def save_model_to_hf(self):
        """
        save the local model to safetensors local directory
        so that vllm can load it

        """

        raise NotImplementedError()

    def launch_vllm(self):
        # TODO:
        # after we haev saved our own model
        # use flag vllm serve <path_to_model>
        # https://discuss.vllm.ai/t/how-to-use-local-model-when-using-vllm-serve/1149

        # args: https://docs.vllm.ai/en/v0.5.4/models/engine_args.html
        command = [
            "vllm",
            "serve",
            # variable args
            "--data-parallel-size",
            str(self.vllm_config.data_parallel_size),
            "--tensor-parallel-size",
            str(self.vllm_config.tensor_parallel_size),
            "--max-num-seqs",
            str(self.vllm_config.max_sequences),
            "--max-num-batched-tokens",
            str(self.vllm_config.max_batched_tokens),
            "--dtype",
            self.vllm_config.dtype,
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

        self.vllm_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL if not self.config.debug else None,
        )
        start = time.perf_counter()
        while not ping_server(IP, PORT):
            logger.info("Waiting for vLLM server to be ready...")
            time.sleep(5)
        end = time.perf_counter()
        logger.info(f"vLLM server is ready in {end - start:.2f} seconds.")

    def launch_eval(self):
        log_file_template = hash_dictConfig(self.config)
        log_path = os.path.join(os.path.abspath(EVAL_LOG_DIR), log_file_template)

        if not os.path.exists(log_path):
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
            log_files = [f"{log_path}/{f}" for f in os.listdir(log_path) if f.endswith(".eval")]
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

    def cleanup(self):
        logger.info("Cleaning up...")
        self.vllm_proess = terminate_process(self.vllm_process, "vLLM")

    def run_evaluation(self):
        try:
            # TODO:
            # self.load_model()
            # self.save_model_to_hf()
            self.launch_vllm()
            self.launch_eval()
        finally:
            self.cleanup()
