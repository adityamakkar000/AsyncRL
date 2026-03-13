import asyncio

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from stax import staxLogger as logger

from src.data.rejection_sample import RejectionSample, rejectionSamplingConfig

cs = ConfigStore.instance()
cs.store(name="base", node=rejectionSamplingConfig)


@hydra.main(version_base=None, config_path="./configs/rejection_sample_config", config_name="main")
def main(cfg: DictConfig) -> None:
    logger.info(f"Rejection Sampling Configuration: \n{OmegaConf.to_yaml(cfg)}")

    rejection_sample = RejectionSample(config=cfg)
    try:
        asyncio.run(rejection_sample.run())
    finally:
        rejection_sample.cleanup()


if __name__ == "__main__":
    main()
