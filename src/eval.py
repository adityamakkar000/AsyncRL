import hydra
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.evals.config import evalConfig
from src.evals.eval import EvalRunner

cs = ConfigStore.instance()
cs.store(name="base", node=evalConfig)


@hydra.main(version_base=None, config_path="./configs/eval")
def main(cfg: DictConfig) -> None:
    logger.info("Evaluation Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    eval_runner = EvalRunner(config=cfg)
    eval_runner.run_evaluation()


if __name__ == "__main__":
    main()
