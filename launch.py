import src.scripts.premption_tooling as TPUJOB

# max_seq_len = [4096, 8192]
# batch_size = [512, 1024]
# grad_accum_steps = [8, 16]
# group_size = [16, 16]
# max_decode_batch_size = [64, 64]

# RUN: TPUJOB.Cross | TPUJOB.Zip | TPUJOB.Vals = TPUJOB.Cross(
#     [
#         TPUJOB.Vals("loss_config.inference_config.max_seq_len", max_seq_len),
#         TPUJOB.Zip(
#             [
#                 TPUJOB.Vals("data_config.batch_size", batch_size),
#                 TPUJOB.Vals("grad_accum_steps", grad_accum_steps),
#                 TPUJOB.Vals("loss_config.inference_config.group_size", group_size),
#                 TPUJOB.Vals("loss_config.inference_config._max_decode_batch_size", max_decode_batch_size),
#             ]
#         ),
#     ]
# )


lag: list[int] = [0, 1, 2, 4, 8, 12]
zero_variance_filter: list[bool] = [True, False]

RUN: TPUJOB.Cross | TPUJOB.Zip | TPUJOB.Vals = TPUJOB.Cross(
    [
        TPUJOB.Vals("async_config.max_lag", lag),
        TPUJOB.Vals("loss_config.rl_config.filter_zero_variance", zero_variance_filter),
    ]
)

BASE_CONFIG = "debug"

job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="debug_lag_filter",
    FIXED_OVERRIDES={
        # "async_config.train_workers": 2,
        # "async_config.max_lag": 2,
        "wandb_config.project": "debug",
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
