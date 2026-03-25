import jax

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


set_jax_cache("./jax/cache")

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
bs = 16
r = 4
config = InferenceConfig(
    temperature=0.7,
    top_p=None,
    top_k=None,
    max_seq_len=4096,
    intial_sequence_len=64,
    max_prefill_sequence_len=256,
    batch_size=bs,
    n_replicas=r,
    group_size=8,
    _max_decode_prompts=bs,
    _max_decode_batch_size=64,
    kv_cache_dtype="bfloat16",
    precompile=True,
    reasoning_budget=None,
    think_mode=True,
    system_prompt=True,
)

engine = InferenceEngine(model, params, config)

key = jax.random.PRNGKey(2303)

tokenizer_inp = [
    r"""Patrick started walking at a constant speed along a straight road from his school to the park. One hour after Patrick left, Tanya started running at a constant speed of $2$ miles per hour faster than Patrick walked, following the same straight road from the school to the park. One hour after Tanya left, Jose started bicycling at a constant speed of $7$ miles per hour faster than Tanya ran, following the same straight road from the school to the park. All three people arrived at the park at the same time. The distance from the school to the park is $\frac{m}{n}$ miles, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.""",
    r"""Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$""",
    r"""A hemisphere with radius $200$ sits on top of a horizontal circular disk with radius $200$, and the hemisphere and disk have the same center. Let $\mathcal T$ be the region of points $P$ in the disk such that a sphere of radius 42 can be placed on top of the disk at $P$ and lie completely inside the hemisphere. The area of $\mathcal T$ divided by the area of the disk is $\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$. """,
    r"""A plane contains points $A$ and $B$ with $AB = 1$. Point $A$ is rotated in the plane counterclockwise through an acute angle $\theta$ around point $B$ to a point $A \prime$. Point $B$ is rotated across a angle of $\theta$ around point $A \prime$ clockwise to a point $B \prime$. $A B \prime = \frac {4}{3}$. If $\cos \theta = \frac{m}{n}$ where $m$ and $n$ are relatively prime positive integers, find $m+n$. """,
] * 4

engine(tokenizer_inp, key, params)

# with stax.Tracker(trace="./profile"):
output: InferenceResults = engine(tokenizer_inp, key, params)

print(output.output_strs)
for key in output.metrics:
    print(f"{key}: {output.metrics[key]}")
breakpoint()

# tps k = 50 : 904.6337280273438
# tps k = None :
