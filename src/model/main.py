from jax.typing import Array
from stax.model_module import HFMixin, mainMixin

# in this file, we want to create a main model class that can
# - instantiate the model
# - load weights from huggingface
# - ckpt/save weights to gcp
# - wrap a method around a model.apply
# - init the KV cache


class mainModel(mainMixin, HFMixin):
    def __init__(self, config, sequence_lengths: Array):
        super().__init__(config)

    def load_from_hf(self, model_name: str):
        pass
