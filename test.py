# from scipy.constants import k
# import os

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

# import jax
# import jax.numpy as jnp

# from einops import rearrange


# def print_sharding(x):
#     jax.debug.visualize_array_sharding(x)


# P = jax.P

# devices = jax.devices()
# mesh = jax.make_mesh((len(devices),), ("dp",))

# jax.set_mesh(mesh)

# grad_accum = 4
# ppo_minibatch_size = 32
# train_batch_size = 256
# T = 8
# ppo_k_steps = train_batch_size // ppo_minibatch_size

# print(f"train_batch_size: {train_batch_size}, ppo_minibatch_size: {ppo_minibatch_size}, ppo_k_steps: {ppo_k_steps}")
# x = jnp.tile(jnp.arange(train_batch_size)[:, None], (1, T))

# x_s = jax.device_put(x, jax.sharding.NamedSharding(mesh, P("dp")))

# x_s_r = rearrange(
#     x_s, "(b g k) t -> k g b t",
#     b=ppo_minibatch_size // grad_accum,
#     g=grad_accum,
#     k=ppo_k_steps,
# )

# breakpoint()

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="mps")

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print("STARTING GENERATIONS NOW")
# conduct text completion
generated_ids = model.generate(**model_inputs, max_new_tokens=100)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
