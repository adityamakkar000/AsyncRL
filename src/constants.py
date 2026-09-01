import os
import queue
import socket
import threading
import time
from dataclasses import dataclass
from multiprocessing.managers import BaseManager

import jax
from jax._src.clusters.cloud_tpu_cluster import GceTpuCluster
from stax.logger import staxLogger as logger

GS_BUCKET = "gs://arl-experiments"
CHECKPOINTS = "checkpoints"
DATA = "data"
PROFILE = "profile"

SERVED_MODEL_NAME = "eval_model"
IP = "localhost"
PORT = "8000"
GPU_MEMORY_UTILIZATION = "0.98"
EVAL_LOG_DIR = "logs/eval_logs"
DISPLAY = "plain"
MAX_TASKS = "5"
VLLM_SERVER_TIMEOUT = 180
OPENAI_TIMEOUT = 1000000

HF_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_params")

INTERUPT_THINKING_PHARSE = "Okay, time is up. Let me stop thinking and formulate a final answer now. \n\n</think>"

SYSTEM_PROMPT = r"""Your task is to follow a systematic, thorough reasoning process before providing the final solution. 
This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. 
Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: \"<think>\n {thoughts} </think>\n\". 
Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas.
After \"</think>\n,\" in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section."""


class QueueManager(BaseManager):
    pass


PORT = 9000
KEY = b"aaa"
VM_IP = socket.gethostbyname(socket.gethostname())
GLOBAL_IP = GceTpuCluster.get_coordinator_address(60).split(":")[0]
TIMEOUT = 60 * 10  # 10 minutes for all queue


class MPQueues:
    prompt_queue: queue.Queue
    rollout_queue: queue.Queue
    weight_sync_queue: queue.Queue[str]
    inference_metrics_queue: queue.Queue[dict]
    eval_prompt_queue: queue.Queue
    eval_rollout_queue: queue.Queue
    eval_done_queue: queue.Queue[str]

    NAMES = (
        "prompt_queue",
        "rollout_queue",
        "weight_sync_queue",
        "inference_metrics_queue",
        "eval_prompt_queue",
        "eval_rollout_queue",
        "eval_done_queue",
    )

    def __init__(self, global_ip: str = GLOBAL_IP, port: int = PORT, key: bytes = KEY):
        self.global_ip = global_ip
        self.port = port
        self.key = key

    def register(self, maxsizes: dict[str, int] | None = None) -> None:
        maxsizes = maxsizes or {}
        for name in self.NAMES:
            local_queue = queue.Queue(maxsize=maxsizes.get(name, 0))
            QueueManager.register(f"get_{name}", callable=lambda q=local_queue: q)

    def start_server(self) -> None:
        def _serve():
            manager = QueueManager(address=("0.0.0.0", self.port), authkey=self.key)
            server = manager.get_server()
            logger.info(f"[Server] Queue server listening on {self.global_ip}:{self.port}...")
            server.serve_forever()

        threading.Thread(target=_serve, daemon=True).start()

    def connect(self, retries: int = 6):
        manager = QueueManager(address=(self.global_ip, self.port), authkey=self.key)

        for _ in range(retries):
            try:
                manager.connect()
                for name in self.NAMES:
                    setattr(self, name, getattr(manager, f"get_{name}")())
                return
            except ConnectionError:
                logger.info(f"[Client] Waiting for server at {self.global_ip}...", log_for_all=True)
                time.sleep(1)

        raise ConnectionError(f"Could not connect to server at {self.global_ip} after {retries} attempts.")


@dataclass
class AsyncOptions:
    train_workers: int
    inference_workers: int
    queues: MPQueues
    train_mesh: jax.sharding.Mesh
    inference_mesh: jax.sharding.Mesh
    global_mesh: jax.sharding.Mesh
