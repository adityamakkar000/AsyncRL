from .config import vLLMConfig


class VLLMEngine:
    def __init__(self, config: vLLMConfig):
        self.config = config
        self.check_config()

    def check_config(self):
        if isinstance(self.config.max_batched_tokens, str):
            if self.config.max_batched_tokens != "auto":
                raise ValueError("If max_batched_tokens is a string, it must be 'auto'.")

    def setup_model(self):
        pass

    def generate(self, prompt):
        # Implement the logic to generate text based on the prompt
        return f"Generated text for prompt: {prompt}"

    def cleanup(self):
        # Implement any necessary cleanup logic
        pass
