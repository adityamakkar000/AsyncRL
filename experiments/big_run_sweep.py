import src.scripts.premption_tooling as TPUJOB

BASE_CONFIG = "big_run"
version = 2

RUN = TPUJOB.Cross(
    [
        TPUJOB.Vals("loss_config/rl_config", ["cispo", "drgrpo"]),
        TPUJOB.Vals("learning_rate_peak", [1e-7, 4e-7, 6e-7]),
    ]
)

job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX=f"llama_sweep_v{version}_2048",
    FIXED_OVERRIDES={
        "wandb_config.project": "big_run",
        "loss_config.inference_config.max_seq_len": 2048,
        "loss_config.inference_config.group_size": 16,
        "loss_config.inference_config.max_decode_batch_size": 48,
        "train_dataset_config.filters.0.max_length": 1000,
        "async_config.train_workers": 2,
        "num_steps": 10000,
        "checkpoint_interval": 50,
        "keep_every": 100,
        "max_checkpoints_to_keep": 1,
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
