import hydra
import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.model import Model, ModelConfig, Qwen3Model

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

    x = jax.random.randint(jax.random.key(32), (4, 5), minval=0, maxval=10000, dtype=jnp.int32)
    seq_lens = jnp.array([2,3,4,5])
    kv_cache = model.init_kv_cache(x)


    logits, cache = model.apply({"params": params}, x=x, sequence_lens=seq_lens)

    breakpoint()

if __name__ == "__main__":
    main()
