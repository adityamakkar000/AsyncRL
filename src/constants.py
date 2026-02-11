import os

GS_BUCKET = "gs://arl-experiments"  # main gs bucket
CHECKPOINTS = "checkpoints"  # checkpoints bucket
DATA = "data"  # data bucket
# DATASETS, HF_PATHS, COLUMNS correspond to each other in index order
DATASETS = ["omnimath", "acereason", "openr1", "numina-cot"]  # dataset bucket names
HF_PATHS = ["KbsdJames/Omni-MATH", "nvidia/AceReason-Math", "open-r1/OpenR1-Math-220k", "numina-cot"]  # hf paths
COLUMNS = [["problem", "solution", "answer"], ["problem", "answer"], ["problem", "solution", "answer", "messages"], ["problem", "solution", "messages"]]  # columns to keep
CACHE = "cache"  # cache bucket


# eval constants
SERVED_MODEL_NAME = "eval_model"
IP = "localhost"
PORT = "8000"
GPU_MEMORY_UTILIZATION = "0.98"
EVAL_LOG_DIR = "logs/eval_logs"
DISPLAY = "plain"
MAX_TASKS = "5"
VLLM_SERVER_TIMEOUT = 180

# use absolute path since VLLM requires a full abspath
HF_CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_params")
