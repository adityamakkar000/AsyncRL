import src.scripts.premption_tooling as TPUJOB

BASE_CONFIG = "main"
NAME = "eval_gsm8k"
RUN = TPUJOB.Cross()

job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    JOB_TYPE=TPUJOB.JOB_TYPES.EVAL,
    EXPERIMENT_PREFIX=NAME,
    FIXED_OVERRIDES={"tasks": ["gsm8k"]},
    BASE_CONFIG=BASE_CONFIG,
    ZONE=TPUJOB.Zone.EUROPE_WEST4_A,
    TPU_TYPE=TPUJOB.TPUType.V6E_8,
    RUNTIME=TPUJOB.Runtime.V2_ALPHA_TPUV6E,
    RETRIES=5,
)

if __name__ == "__main__":
    TPUJOB.launch(job)
