import src.scripts.premption_tooling as TPUJOB

BASE_CONFIG = "big_run"
version = 1

RUN = TPUJOB.Zip(
    [
        TPUJOB.Vals(
            "model_config",
            ["qwen_1.7b_base", "qwen_4b_base", "qwen_8b_base", "llama_3.1_8b_instruct", "llama_3.1_8b_instruct"],
        ),
        TPUJOB.Vals("loss_config/rl_config", ["cispo", "cispo", "cispo", "drgrpo", "cispo"]),
        TPUJOB.Vals("learning_rate_peak", [1e-6, 1e-6, 1e-6, 4e-7, 4e-7]),
        TPUJOB.Vals("loss_config.inference_config.max_decode_batch_size", [88, 56, 48, 48, 48]),
    ]
)

job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX=f"final_v{version}_4096",
    FIXED_OVERRIDES={
        "wandb_config.project": "big_run",
        "loss_config.inference_config.max_seq_len": 4096,
        "loss_config.inference_config.group_size": 16,
        "train_dataset_config.filters.0.max_length": 1000,
        "async_config.train_workers": 4,
        "grad_accum_steps": 8,
        "num_steps": 10000,
        "checkpoint_interval": 50,
        "keep_every": 100,
        "max_checkpoints_to_keep": 1,
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
    TPUJOB.launch(job)
