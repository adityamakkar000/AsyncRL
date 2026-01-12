import os

import hydra
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.trainer import Trainer, TrainerConfig

cs = ConfigStore.instance()
cs.store(name="base_train_config", node=TrainerConfig)

@hydra.main(version_base=None, config_path="./configs/train")
def main(cfg: DictConfig) -> None:
    logger.info("Training Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    xla_flags_str = " --".join([f"{flag.name}={flag.value}" for flag in cfg.xla_flags]) 
    logger.info(f"Setting XLA flags: {xla_flags_str}")
    os.environ["XLA_FLAGS"] = xla_flags_str

    trainer = Trainer(config=cfg)
    trainer.train()

if __name__ == "__main__":
    main()
