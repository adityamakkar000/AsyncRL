import src.scripts.premption_tooling as TPUJOB

max_seq_len = [4096]
batch_size = [512]
grad_accum_steps = [8]
group_size = [16]
max_prompt_decode = [1, 2, 4, 8, 16, 32, 64]
max_decode_batch_size = [i * 16 // 2 for i in max_prompt_decode]


RUN: TPUJOB.Cross | TPUJOB.Zip | TPUJOB.Vals = TPUJOB.Cross(
    [
        TPUJOB.Vals("loss_config.inference_config.max_seq_len", max_seq_len),
        TPUJOB.Vals("data_config.batch_size", batch_size),
        TPUJOB.Vals("grad_accum_steps", grad_accum_steps),
        TPUJOB.Vals("loss_config.inference_config.group_size", group_size),
        TPUJOB.Zip(
            [
                TPUJOB.Vals("loss_config.inference_config.max_decode_batch_size", max_decode_batch_size),
                TPUJOB.Vals("loss_config.inference_config.max_decode_prompts", max_prompt_decode),
            ]
        ),
    ]
)


BASE_CONFIG = "baseline"

job = TPUJOB.LAUNCH_JOB(
    RUN=RUN,
    EXPERIMENT_PREFIX="test",
    FIXED_OVERRIDES={"wandb_config.project": "inference_test", "num_steps": 10},
    BASE_CONFIG=BASE_CONFIG,
    ZONE=TPUJOB.Zone.US_EAST5_A,
    JOB_TYPE=TPUJOB.JOB_TYPES.TRAIN,
    TPU_TYPE=TPUJOB.TPUType.V5P_32,
    RUNTIME=TPUJOB.Runtime.V2_ALPHA_TPUV5,
    RETRIES=5,
)

if __name__ == "__main__":
    TPUJOB.launch(job)
