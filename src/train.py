import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from stax import staxLogger as logger

from src.trainer import Trainer, TrainerConfig

cs = ConfigStore.instance()
cs.store(name="base", node=TrainerConfig)


@hydra.main(version_base=None, config_path="./configs/train")
def main(cfg: DictConfig) -> None:
    logger.info("Training Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    try:
        trainer = Trainer(config=cfg)
        trainer.train()
    finally:
        trainer.finish()


if __name__ == "__main__":
    main()
