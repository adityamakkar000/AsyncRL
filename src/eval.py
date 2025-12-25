import hydra
from src.evals.eval import EvalRunner
from src.evals.config import evalConfig
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, MISSING
from loguru import logger

cs = ConfigStore.instance()
cs.store(name="base_eval_config", node=evalConfig)


@hydra.main(version_base=None, config_path="./configs/eval")
def main(cfg: DictConfig) -> None:
    logger.info("Evaluation Configuration:")
    logger.info(f"\n" + OmegaConf.to_yaml(cfg))

    eval_runner = EvalRunner(config=cfg)
    eval_runner.run_evaluation()


if __name__ == "__main__":
    main()
