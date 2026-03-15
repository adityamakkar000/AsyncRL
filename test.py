import numpy as np
import torch
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B-Base"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model.train()

input_ids = torch.tensor(np.arange(128)).unsqueeze(0).to(model.device)  # shape: (1, 10)

with torch.no_grad():
    outputs = model(input_ids=input_ids)

logits = outputs.logits  # shape: (1, 10, vocab_size)
print("Logits shape:", logits.shape)
print("Logits:\n", logits)


for i in range(10):
    print(i)
