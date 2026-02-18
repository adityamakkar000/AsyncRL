import hydra
import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.model import Model, ModelConfig

jax.config.update("jax_default_matmul_precision", "highest")
jax.numpy.set_printoptions(precision=9)

cs = ConfigStore.instance()
cs.store(name="base_model_config", node=ModelConfig)


@hydra.main(version_base=None, config_path="./configs/train/model_config/")
def main(cfg: DictConfig) -> None:
    logger.info("Model Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    model = Model(cfg)
    rng = jax.random.PRNGKey(0)
    out_state = model.init_state(rng, None, sharding=None, abstract=False)
    params = out_state["params"]

    x = jax.random.randint(jax.random.key(32), (4, 128), minval=0, maxval=10000, dtype=jnp.int32)
    seq_lens = jnp.array([64, 79, 112, 128])
    cache = model.init_kv_cache(4, None, dtype="float32")

    out1, cache1 = model.apply({"params": params}, x=x, sequence_lens=seq_lens, kv_cache=None)
    out2, cache2 = model.apply({"params": params}, x=x, sequence_lens=seq_lens, kv_cache=cache, attention_len=128)

    def make_prompt_mask(max_seq_len: int, cache_len, seq_lens):
        """
        This function generates a boolean mask that identifies valid (non-padded) tokens
        within the cache region of each sequence. It handles left-padded sequences by
        masking out padding tokens at the beginning of each sequence.

            max_seq_len (int): The maximum sequence length including any tokens beyond the cache.
            cache_len (int): The length of the cached tokens (KV cache size).
            seq_lens (Array): Array of shape (batch_size,) containing the actual
                sequence lengths for each batch element. Each element should be <= cache_len.

        Returns:
            Array: A boolean mask of shape (batch_size, max_seq_len) where True indicates
                valid (non-padded) tokens within the cache region, and False indicates either
                padding tokens or positions beyond the cache.

        Example:
            >>> seq_lens = jnp.array([3, 5])
            >>> max_seq_len = 5
            >>> cache_len = 4
            >>> make_prompt_mask(max_seq_len, cache_len, seq_lens)
            # Returns:
            # [[False, False, True, True, False],
            #  [True,  True,  True, True, False]]
            #
            # First sequence: 3 valid tokens, left-padded with 1 token, 1 position beyond cache
            # Second sequence: 4 valid tokens (capped by cache_len), 1 position beyond cache
        """
        raw_length = jnp.arange(max_seq_len)[None, :]
        # left padding mask
        padding_mask = raw_length >= (cache_len - seq_lens[:, None])
        # cache mask
        cache_mask = raw_length < cache_len
        return padding_mask & cache_mask

    _mask = make_prompt_mask(128, 128, seq_lens)[:, :, None]

    breakpoint()


if __name__ == "__main__":
    main()
