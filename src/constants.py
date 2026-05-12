import os
import queue
import socket
from dataclasses import dataclass
from multiprocessing.managers import BaseManager

import jax
from jax._src.clusters.cloud_tpu_cluster import GceTpuCluster

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


class QueueManager(BaseManager):
    pass


PORT = 9000
KEY = b"aaa"
VM_IP = socket.gethostbyname(socket.gethostname())
GLOBAL_IP = GceTpuCluster.get_coordinator_address(60).split(":")[0]


@dataclass
class AsyncOptions:
    train_workers: int
    inference_workers: int
    prompt_queue: queue.Queue
    rollout_queue: queue.Queue
    weight_sync_queue: queue.Queue[str]
    inference_metrics_queue: queue.Queue[dict]
    train_mesh: jax.sharding.Mesh
    inference_mesh: jax.sharding.Mesh
