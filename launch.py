import src.scripts.premption_tooling as TPUJOB

# max_seq_len = [4096, 8192]
# batch_size = [1024, 1024]
# group_size = [8, 16]


RUN: TPUJOB.Vals | TPUJOB.Zip | TPUJOB.Cross = TPUJOB.Zip(
    [
        TPUJOB.Vals("loss_config.annealing_config.use_annealing", [0]),
    ]
)

# RUN: Cross | Zip | Vals = Cross(
#     [
#         Vals("loss_config.inference_config.max_seq_len", max_seq_len),
#         Zip(
#             [
#                 Vals("data_config.train_config.batch_size", batch_size),
#                 Vals("data_config.val_config.batch_size", batch_size),
#                 Vals("loss_config.inference_config.group_size", group_size),
#             ]
#         ),
#     ]
# )


BASE_CONFIG = "debug"
# RUN = TPUJOB.Cross()


job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="debug",
    FIXED_OVERRIDES=dict(),
    BASE_CONFIG=BASE_CONFIG,
    ZONE=TPUJOB.Zone.US_EAST5_A,
    JOB_TYPE=TPUJOB.JOB_TYPES.TRAIN,
    TPU_TYPE=TPUJOB.TPUType.V5P_32,
    RUNTIME=TPUJOB.Runtime.V2_ALPHA_TPUV5,
    RETRIES=5,
)

if __name__ == "__main__":
    TPUJOB.launch(job)
