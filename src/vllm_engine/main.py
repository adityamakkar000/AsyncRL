import os
import subprocess
import time

from loguru import logger

from src.constants import GPU_MEMORY_UTILIZATION, IP, PORT, SERVED_MODEL_NAME, VLLM_SERVER_TIMEOUT

from .config import vLLMConfig
from .utils import format_command, ping_server, terminate_process


class vLLMEngine:
    def __init__(self, config: vLLMConfig, debug: bool = False):
        self.config = config
        self.debug = debug
        self.vllm_process = None
        self.check_config()

    def check_config(self):
        if isinstance(self.config.max_batched_tokens, str):
            if self.config.max_batched_tokens != "auto":
                raise ValueError("If max_batched_tokens is a string, it must be 'auto'.")

    def launch_vllm(self, model_path: str):
        """Launch the vLLM server with the specified model path. The model path should be a local path when used with evals and model name when used with Rejection Sampling."""
        command = [
            "vllm",
            "serve",
            model_path,
            # variable args
            "--data-parallel-size",
            str(self.config.data_parallel_size),
            "--tensor-parallel-size",
            str(self.config.tensor_parallel_size),
            "--max-num-seqs",
            str(self.config.max_sequences),
            "--max-num-batched-tokens",
            str(self.config.max_batched_tokens),
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
            stdout=subprocess.DEVNULL if not self.debug else None,
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

    def cleanup(self):
        logger.info("Cleaning up...")
        self.vllm_process = terminate_process(self.vllm_process, "vLLM")
