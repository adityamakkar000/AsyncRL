import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from stax import staxLogger as logger

from src.data.process_datasets import ProcessDataset
from src.data.config import DataConfig


cs = ConfigStore.instance()
cs.store(name="base", node=DataConfig)


@hydra.main(version_base=None, config_path="./configs/process_datasets", config_name="main")
def main(cfg: DictConfig) -> None:
    """Entry point for Hydra."""
    logger.info("ProcessDataset configuration:\n" + OmegaConf.to_yaml(cfg))
    processor = ProcessDataset(cfg)
    processor._process_and_upload()


if __name__ == "__main__":
    main()