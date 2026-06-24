import src.scripts.premption_tooling as TPUJOB

max_seq_len = [4096]

lr = [1e-6, 5e-7]

batch_size = [512, 768]
grad_accum_steps = [4, 8]

variance_filtering = [True, False]

RUN: TPUJOB.Cross | TPUJOB.Zip | TPUJOB.Vals = TPUJOB.Cross(
    [
        TPUJOB.Vals("loss_config.inference_config.max_seq_len", max_seq_len),
        TPUJOB.Zip(
            [
                TPUJOB.Vals("learning_rate_peak", lr),
                TPUJOB.Vals("learning_rate_end", lr),
            ]
        ),
        TPUJOB.Zip(
            [
                TPUJOB.Vals("data_config.batch_size", batch_size),
                TPUJOB.Vals("grad_accum_steps", grad_accum_steps),
            ]
        ),
        TPUJOB.Vals("loss_config.rl_config.filter_zero_variance", variance_filtering),
    ]
)

BASE_CONFIG = "baseline"

job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="async_baseline_inf_fix",
    FIXED_OVERRIDES={
        "wandb_config.project": "baseline_v6",
        "learning_rate_init": 0.0,
        "warmup_steps": 0.03,
        "decay_steps": 0.9,
        "loss_config.inference_config.max_decode_prompts": 8,
        "loss_config.inference_config.max_decode_batch_size": 64,
        "loss_config.inference_config.group_size": 16,
        "num_steps": 3000,
        "log_generations_every_n_steps": 100,
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
