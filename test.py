import glob

from safetensors import safe_open

for name in ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]:
    all_keys = []
    for file in glob.glob(name + "/*safetensors"):
        with safe_open(file, framework="torch") as f:
            all_keys.extend(f.keys())
    print(f"{name}: lm_head.weight present = {'lm_head.weight' in all_keys}")
