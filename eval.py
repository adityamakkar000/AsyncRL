import src.scripts.premption_tooling as TPUJOB

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
