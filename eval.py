import src.scripts.premption_tooling as TPUJOB

max_seq_len = [4096, 8192]
batch_size = [1024, 1024]
group_size = [8, 16]

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


BASE_CONFIG = "main"
NAME = "eval_gsm8k"
RUN = TPUJOB.Cross()
TYPE = TPUJOB.JOB_TYPE.EVAL

job = TPUJOB.LaunchJob(
    run=RUN,
    job_type=TYPE,
    experiment_prefix=NAME,
    fixed_overrides={"tasks": ["gsm8k"]},
    base_config=BASE_CONFIG,
    zone=TPUJOB.Zone.EUROPE_WEST4_A,
    tpu_type=TPUJOB.TPUType.V6E_8,
    runtime=TPUJOB.Runtime.V2_ALPHA_TPUV6E,
    retries=5,
)

if __name__ == "__main__":
    TPUJOB.launch(job)
