import asyncio
import os
import subprocess
import time

from loguru import logger
from openai import AsyncOpenAI
from PIL.ImImagePlugin import OPEN

from src.constants import GPU_MEMORY_UTILIZATION, IP, OPENAI_TIMEOUT, PORT, SERVED_MODEL_NAME, VLLM_SERVER_TIMEOUT

from .config import SamplingParams, vLLMConfig, vLLMOutput
from .utils import format_command, ping_server, terminate_process


class vLLMEngine:
    def __init__(self, config: vLLMConfig, max_workers: int, debug: bool = False):
        self.config = config
        self.debug = debug
        self.vllm_process = None
        self.client = AsyncOpenAI(
            base_url=f"http://{IP}:{PORT}/v1",
            api_key="",
            timeout=OPENAI_TIMEOUT,
        )
        self.check_config()
        self.rejection_semaphore = asyncio.Semaphore(max_workers)

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

    async def call_vllm(self, prompt: str, sampling_config: SamplingParams) -> list[str]:
        """Calls the vLLM server to generate completions for a given prompt."""
        async with self.rejection_semaphore:
            response = await self.client.completions.create(
                model=SERVED_MODEL_NAME,
                prompt=prompt,
                max_tokens=sampling_config.max_sequence_len,
                temperature=sampling_config.temperature,
                top_p=sampling_config.top_p,
                n=sampling_config.pass_at,
            )
            return [choice.text for choice in response.choices]

    async def await_completions(self, prompts: list[str], sampling_config: SamplingParams) -> list[list[str]]:
        """Synchronous wrapper around call_vllm."""

        tasks = [self.call_vllm(prompt, sampling_config) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def generate_completions(
        self, prompts: list[str], max_sequence_len: int, pass_at: int, temperature: float, top_p: float
    ) -> vLLMOutput:
        sampling_config = SamplingParams(
            max_sequence_len=max_sequence_len,
            pass_at=pass_at,
            temperature=temperature,
            top_p=top_p,
        )
        logger.info(f"Generating completions with sampling config: {sampling_config}")
        completion_outputs = asyncio.run(self.await_completions(prompts, sampling_config))

        return vLLMOutput(prompts=prompts, completions=completion_outputs)

    def cleanup(self):
        logger.info("Cleaning up...")
        self.vllm_process = terminate_process(self.vllm_process, "vLLM")
