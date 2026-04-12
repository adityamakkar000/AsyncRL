import jax

from src.model import Model, ModelConfig, QwenConfig

from .config import InferenceConfig, InferenceResults
from .main import InferenceEngine


def set_jax_cache(path: str, log_compile: bool = False):
    """Sets the JAX cache directory to the specified path."""
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    jax.config.update("jax_log_compiles", log_compile)


set_jax_cache("./jax/cache", log_compile=False)

model_config = ModelConfig(
    "Qwen/Qwen3-0.6B",
    qwen_config=QwenConfig(
        vocab_size=151_936,
        d_ff=3072,
        sequence_len=16384,
        model_dim=1024,
        n_heads=16,
        n_groups=8,
        n_layers=28,
        head_dim=128,
        rope_base=1_000_000,
        activation_dtype="bfloat16",
    ),
)
model = Model(model_config)

params = model.init_state(jax.random.PRNGKey(0), None, abstract=False)
config = InferenceConfig(
    temperature=0.7,
    top_p=None,
    top_k=None,
    max_seq_len=4096,
    initial_sequence_len=64,
    max_prefill_sequence_len=256,
    n_replicas=4,
    group_size=8,
    _max_decode_prompts=1,
    _max_decode_batch_size=8,
    kv_cache_dtype="bfloat16",
    reasoning_budget=None,
    think_mode=True,
    system_prompt=True,
)

engine = InferenceEngine(model, params, config)

key = jax.random.PRNGKey(2303)

tokenizer_inp = [
    r"""Patrick started walking at a constant speed along a straight road from his school to the park. One hour after Patrick left, Tanya started running at a constant speed of $2$ miles per hour faster than Patrick walked, following the same straight road from the school to the park. One hour after Tanya left, Jose started bicycling at a constant speed of $7$ miles per hour faster than Tanya ran, following the same straight road from the school to the park. All three people arrived at the park at the same time. The distance from the school to the park is $\frac{m}{n}$ miles, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.""",
]

engine(tokenizer_inp, key, params)
output: InferenceResults = engine(tokenizer_inp, key, params)

# print(output.output_strs)
for key in output.metrics:
    print(f"{key}: {output.metrics[key]}")
breakpoint()
