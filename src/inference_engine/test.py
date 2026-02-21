import jax

from src.constants import CACHE, GS_BUCKET
from src.model import Model, ModelConfig, QwenConfig

from .config import InferenceConfig, InferenceResults
from .main import InferenceEngine


def set_jax_cache(path: str):
    """Sets the JAX cache directory to the specified path."""
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    # jax.config.update("jax_log_compiles", True)


cache_path = f"{GS_BUCKET}/{CACHE}"
set_jax_cache(cache_path)


model_config = ModelConfig(
    "Qwen/Qwen3-0.6B",
    qwen_config=QwenConfig(
        vocab_size=151_936,
        d_ff=3072,
        sequence_len=16_384,
        model_dim=1024,
        n_heads=16,
        n_groups=8,
        head_dim=128,
        n_layers=28,
        rope_base=1_000_000,
        activation_dtype="bfloat16",
    ),
)
model = Model(model_config)

params = model.init_state(jax.random.PRNGKey(0), None, abstract=False)
bs = 1
r = 4
config = InferenceConfig(
    temperature=0.6,
    top_p=0.95,
    top_k=50,
    max_seq_len=256,
    intial_sequence_len=64,
    max_prefill_sequence_len=256,
    batch_size=bs,
    n_replicas=r,
    group_size=bs * r,
    kv_cache_dtype="bfloat16",
    precompile=True,
    params_dtype="float32",
    reasoning_budget=64,
)

engine = InferenceEngine(model, params, config)

key = jax.random.PRNGKey(2303)

tokenizer_inp = [" Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$"]

output: InferenceResults = engine(tokenizer_inp, key, params)
print(output)

breakpoint()
