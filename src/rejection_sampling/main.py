from loguru import logger

from src.vllm_engine.main import vLLMEngine


class RejectionSample:
    def __init__(self, config):
        self.config = config
        self.vllm_engine = vLLMEngine(config.vllm_config)

    def check_config(self):
        pass

    def upload_dataset(self):
        pass

    def setup(self):
        pass

    def cleanup(self):
        logger.info("Clearning up vLLM engine...")
        self.vllm_engine.cleanup()

    def run(self):
        pass
