from src.scripts.premption_tooling import LAUNCH_JOB, Cross, Runtime, TPUType, Vals, Zip, Zone, launch

lrs = [3e-6, 3e-7]
minbatch_size = [512]
grad_steps = [8]
dataset = ["omnimath_debug_1500_1750"]
algo = ["rloo", "cispo"]

RUN: Cross | Zip | Vals = Cross(
    [
        Zip(
            [
                Vals("learning_rate_peak", lrs),
                Vals("learning_rate_end", lrs),
            ]
        ),
        Zip(
            [
                Vals("loss_config.rl_config.ppo_minibatch_size", minbatch_size),
                Vals("grad_accum_steps", grad_steps),
            ]
        ),
        Vals("loss_config.rl_config.algorithm", algo),
        Vals("data_config/datasets@data_config.train_config", dataset),
    ]
)

job = LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="nothink-v2",
    FIXED_OVERRIDES={
        "wandb_config.project": "AnnealedRL",
    },
    BASE_CONFIG="run1",
    ZONE=Zone.US_EAST5_A,
    TPU_TYPE=TPUType.V5P_32,
    RUNTIME=Runtime.V2_ALPHA_TPUV5,
)

if __name__ == "__main__":
    launch(job)
