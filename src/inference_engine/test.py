import jax

from src.model import Model, ModelConfig, QwenConfig

from .config import InferenceConfig, InferenceResults
from .main import InferenceEngine

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
config = InferenceConfig(
    temperature=0.6,
    top_p=0.95,
    top_k=50,
    max_seq_len=1024,
    intial_sequence_len=16,
    batch_size=1,
    n_replicas=1,
    group_size=1,
    kv_cache_dtype="bfloat16",
    precompile=False,
)

engine = InferenceEngine(model, params, config)

key = jax.random.PRNGKey(2303)
params = model.init_state(jax.random.PRNGKey(0), None, abstract=False)

tokenizer_inp = ["What is 2 + 2"]

engine(tokenizer_inp, key, params, detokenize=True)
output: InferenceResults = engine(tokenizer_inp, key, params, detokenize=True)
print(output)

breakpoint()
