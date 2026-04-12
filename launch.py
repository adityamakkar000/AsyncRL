from src.scripts.premption_tooling import LAUNCH_JOB, Cross, Runtime, TPUType, Vals, Zip, Zone, launch

max_seq_len = [4096, 8192]
batch_size = [1024, 1024]
group_size = [8, 16]

RUN: Cross | Zip | Vals = Cross(
    [
        Vals("loss_config.inference_config.max_seq_len", max_seq_len),
        Zip(
            [
                Vals("data_config.train_config.batch_size", batch_size),
                Vals("data_config.val_config.batch_size", batch_size),
                Vals("loss_config.inference_config.group_size", group_size),
            ]
        ),
    ]
)


job = LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="baselinev1",
    FIXED_OVERRIDES=dict(),
    BASE_CONFIG="baseline",
    ZONE=Zone.US_EAST5_A,
    TPU_TYPE=TPUType.V5P_64,
    RUNTIME=Runtime.V2_ALPHA_TPUV5,
)

if __name__ == "__main__":
    launch(job)
