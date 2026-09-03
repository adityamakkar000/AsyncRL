import src.scripts.premption_tooling as TPUJOB

max_seq_len = [2048]
# variance_filtering = [True, False]
async_workers = [1, 2]

RUN: TPUJOB.Cross | TPUJOB.Zip | TPUJOB.Vals = TPUJOB.Cross(
    [
        TPUJOB.Vals("loss_config.inference_config.max_seq_len", max_seq_len),
        # TPUJOB.Vals("loss_config.filter_zero_variance", variance_filtering),
        TPUJOB.Vals("async_config.train_workers", async_workers),
    ]
)

BASE_CONFIG = "big_run"

job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="llama_big_run_v2",
    FIXED_OVERRIDES={
        "wandb_config.project": "big_run_debug",
        "loss_config.inference_config.max_decode_batch_size": 32,
        "loss_config.inference_config.group_size": 16,
        "num_steps": 10000,
        "checkpoint_interval": 50,
        "keep_every": 100,
        "max_checkpoints_to_keep": 1,
        "model_config": "qwen_0.6b_base",
    },
    BASE_CONFIG=BASE_CONFIG,
    ZONE=TPUJOB.Zone.US_EAST5_A,
    JOB_TYPE=TPUJOB.JOB_TYPES.TRAIN,
    TPU_TYPE=TPUJOB.TPUType.V5P_32,
    RUNTIME=TPUJOB.Runtime.V2_ALPHA_TPUV5,
    RETRIES=5,
    DEBUG=False,
)

if __name__ == "__main__":
    TPUJOB.launch(job)
