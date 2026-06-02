import src.scripts.premption_tooling as TPUJOB

max_seq_len = [4096, 8192]
batch_size = [512, 1024]
grad_accum_steps = [8, 16]
group_size = [16, 16]
max_decode_batch_size = [64, 64]

RUN: TPUJOB.Cross | TPUJOB.Zip | TPUJOB.Vals = TPUJOB.Cross(
    [
        TPUJOB.Vals("loss_config.inference_config.max_seq_len", max_seq_len),
        TPUJOB.Zip(
            [
                TPUJOB.Vals("data_config.batch_size", batch_size),
                TPUJOB.Vals("grad_accum_steps", grad_accum_steps),
                TPUJOB.Vals("loss_config.inference_config.group_size", group_size),
                TPUJOB.Vals("loss_config.inference_config._max_decode_batch_size", max_decode_batch_size),
            ]
        ),
    ]
)

BASE_CONFIG = "baseline"

job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="async_baseline_v5",
    FIXED_OVERRIDES={
        "async_config.train_workers": 2,
        "async_config.max_prompt_queue_size": 4,
        "wandb_config.project": "baseline_fr",
    },
    BASE_CONFIG=BASE_CONFIG,
    ZONE=TPUJOB.Zone.US_EAST5_A,
    JOB_TYPE=TPUJOB.JOB_TYPES.TRAIN,
    TPU_TYPE=TPUJOB.TPUType.V5P_32,
    RUNTIME=TPUJOB.Runtime.V2_ALPHA_TPUV5,
    RETRIES=5,
)

if __name__ == "__main__":
    TPUJOB.launch(job)
