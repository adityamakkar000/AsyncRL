import src.scripts.premption_tooling as TPUJOB

max_decode_batch_size = [8, 16, 32, 64, 128, 256]

RUN: TPUJOB.Cross | TPUJOB.Zip | TPUJOB.Vals = TPUJOB.Cross(
    [
        TPUJOB.Vals("loss_config.inference_config.max_decode_batch_size", max_decode_batch_size),
    ]
)

BASE_CONFIG = "debug"

job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="inf_sweep",
    FIXED_OVERRIDES={
        "wandb_config.project": "inference_sweep",
        "num_steps": 20,
        "eval_config.eval_every_n_steps": 1000,
        "log_generations_every_n_steps": 1000,
    },
    BASE_CONFIG=BASE_CONFIG,
    ZONE=TPUJOB.Zone.US_EAST5_A,
    JOB_TYPE=TPUJOB.JOB_TYPES.TRAIN,
    TPU_TYPE=TPUJOB.TPUType.V5P_16,
    RUNTIME=TPUJOB.Runtime.V2_ALPHA_TPUV5,
    RETRIES=5,
)

if __name__ == "__main__":
    TPUJOB.launch(job)
