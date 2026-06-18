from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
chat = [{"role": "user", "content": "hello how are you"}]


breakpoint()
