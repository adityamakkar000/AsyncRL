import os

# GS_BUCKET = "gs://arl-us-east-5"
GS_BUCKET = "gs://arl-us-central-1"
RUN_PATH = f"{GS_BUCKET}/runs"

HF_CHECKPOINT_PATH = os.path.expanduser("~/hf_params")

INTERUPT_THINKING_PHARSE = "Okay, time is up. Let me stop thinking and formulate a final answer now. \n\n</think>"
