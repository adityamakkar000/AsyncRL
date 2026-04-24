import os

GS_BUCKET = "gs://arl-experiments"  # main gs bucket
CHECKPOINTS = "checkpoints"  # checkpoints bucket
DATA = "data"  # data bucket
CACHE = "cache"  # cache bucket
PROFILE = "profile"

# eval constants
SERVED_MODEL_NAME = "eval_model"
IP = "localhost"
PORT = "8000"
GPU_MEMORY_UTILIZATION = "0.98"
EVAL_LOG_DIR = "logs/eval_logs"
DISPLAY = "plain"
MAX_TASKS = "5"
VLLM_SERVER_TIMEOUT = 180
OPENAI_TIMEOUT = 1000000

# use absolute path since VLLM requires a full abspath
HF_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_params")
