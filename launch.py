import src.scripts.premption_tooling as TPUJOB
from src.scripts.premption_tooling import Vals, Cross, Zip

# max_seq_len = [4096, 8192]
# batch_size = [1024, 1024]
# group_size = [8, 16]


RUN = Zip(
    [
        Vals("loss_config.annealing_config.use_annealing", [1]),
        Vals("loss_config.annealing_config.schedule", ["cosine"]),
        Vals("loss_config.annealing_config.annealing_steps", [0.2])
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


BASE_CONFIG = "debug_anneal"
# RUN = TPUJOB.Cross()


job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="debug",
    FIXED_OVERRIDES=dict(),
    BASE_CONFIG=BASE_CONFIG,
    ZONE=TPUJOB.Zone.US_EAST5_A,
    TPU_TYPE=TPUJOB.TPUType.V5P_32,
    RUNTIME=TPUJOB.Runtime.V2_ALPHA_TPUV5,
    RETRIES=5,
)

if __name__ == "__main__":
    TPUJOB.launch(job)
