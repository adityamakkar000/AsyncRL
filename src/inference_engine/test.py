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
    "Qwen/Qwen3-1.7B",
    qwen_config=QwenConfig(
        vocab_size=151_936,
        d_ff=6144,
        sequence_len=16384,
        model_dim=2048,
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
bs = 1
r = 1
config = InferenceConfig(
    temperature=0.7,
    top_p=None,
    top_k=None,
    max_seq_len=512,
    intial_sequence_len=64,
    max_prefill_sequence_len=256,
    batch_size=bs,
    n_replicas=r,
    group_size=bs * r,
    kv_cache_dtype="bfloat16",
    precompile=False,
    reasoning_budget=None,
    think_mode=True,
    system_prompt=True,
)

engine = InferenceEngine(model, params, config)

key = jax.random.PRNGKey(2303)

tokenizer_inp = [
    r"""Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""
]

# engine(tokenizer_inp, key, params)
output: InferenceResults = engine(tokenizer_inp, key, params)

print(output.output_strs)
for key in output.metrics:
    print(f"{key}: {output.metrics[key]}")
breakpoint()

# tps  k = 50 : 904.6337280273438
# tps k = None :
