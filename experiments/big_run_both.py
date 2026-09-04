import src.scripts.premption_tooling as TPUJOB

BASE_CONFIG = "big_run"

COMMON_OVERRIDES = {
    "wandb_config.project": "big_run",
    "loss_config.inference_config.group_size": 16,
    "train_dataset_config.filters.0.max_length": 1000,
    "num_steps": 10000,
    "checkpoint_interval": 50,
    "keep_every": 100,
    "max_checkpoints_to_keep": 1,
}

job_2048 = TPUJOB.LAUNCH_JOB(
    RUN=TPUJOB.Vals("loss_config.inference_config.max_seq_len", [2048]),
    EXPERIMENT_PREFIX="llama_big_run_v1_2048",
    FIXED_OVERRIDES={
        **COMMON_OVERRIDES,
        "async_config.train_workers": 1,
        "loss_config.inference_config.max_decode_batch_size": 48,
    },
    BASE_CONFIG=BASE_CONFIG,
    ZONE=TPUJOB.Zone.US_EAST5_A,
    JOB_TYPE=TPUJOB.JOB_TYPES.TRAIN,
    TPU_TYPE=TPUJOB.TPUType.V5P_32,
    RUNTIME=TPUJOB.Runtime.V2_ALPHA_TPUV5,
    RETRIES=5,
    DEBUG=False,
)

job_4096 = TPUJOB.LAUNCH_JOB(
    RUN=TPUJOB.Vals("loss_config.inference_config.max_seq_len", [4096]),
    EXPERIMENT_PREFIX="llama_big_run_v1_4096",
    FIXED_OVERRIDES={
        **COMMON_OVERRIDES,
        "async_config.train_workers": 2,
        "loss_config.inference_config.max_decode_batch_size": 24,
        "grad_accum_steps": 8,
    },
    BASE_CONFIG=BASE_CONFIG,
    ZONE=TPUJOB.Zone.US_EAST5_A,
    JOB_TYPE=TPUJOB.JOB_TYPES.TRAIN,
    TPU_TYPE=TPUJOB.TPUType.V5P_64,
    RUNTIME=TPUJOB.Runtime.V2_ALPHA_TPUV5,
    RETRIES=5,
    DEBUG=False,
)

if __name__ == "__main__":
    import sys

    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only in (None, "2048"):
        TPUJOB.launch(job_2048)
    if only in (None, "4096"):
        TPUJOB.launch(job_4096)
