from src.scripts.premption_tooling import LAUNCH_JOB, Cross, Runtime, TPUType, Vals, Zip, Zone, launch

lrs = [3e-6, 1e-6]

RUN: Cross | Zip | Vals = Cross(
    [
        Zip(
            [
                Vals("learning_rate_peak", lrs),
                Vals("learning_rate_end", lrs),
            ]
        ),
    ]
)

job = LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="nothink-v2",
    FIXED_OVERRIDES={
        "wandb_config.project": "Debug",
    },
    BASE_CONFIG="debug",
    ZONE=Zone.US_EAST5_A,
    TPU_TYPE=TPUType.V5P_32,
    RUNTIME=Runtime.V2_ALPHA_TPUV5,
)

if __name__ == "__main__":
    launch(job)
