import src.scripts.premption_tooling as TPUJOB

model_name = [
    "final_run_max_seq_le4096_learning_r1e-06_learning_r1e-06_batch_size768_grad_accum8_filter_zerTrue",
    "final_run_max_seq_le4096_learning_r1e-06_learning_r1e-06_batch_size768_grad_accum8_filter_zerFalse",
]
steps = [
    0,
    30,
    60,
    90,
    120,
    # 150,
    # 180,
    # 210,
    # 240,
    # 270,
    # 300,
]

RUN = TPUJOB.Cross(
    [
        TPUJOB.Vals("model_config.model_name", model_name),
        TPUJOB.Vals("model_config.step_number", steps),
    ]
)
BASE_CONFIG = "main"

job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="evals_final_run",
    FIXED_OVERRIDES=dict(),
    BASE_CONFIG=BASE_CONFIG,
    ZONE=TPUJOB.Zone.EUROPE_WEST4_A,
    JOB_TYPE=TPUJOB.JOB_TYPES.EVAL,
    TPU_TYPE=TPUJOB.TPUType.V6E_8,
    RUNTIME=TPUJOB.Runtime.V2_ALPHA_TPUV6E,
    RETRIES=5,
)

if __name__ == "__main__":
    TPUJOB.launch(job)
