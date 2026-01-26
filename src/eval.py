import os

os.environ["JAX_PLATFORMS"] = "cpu"
import hydra
import jax
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.evals import EvalRunner, evalConfig

cs = ConfigStore.instance()
cs.store(name="base", node=evalConfig)


@hydra.main(version_base=None, config_path="./configs/eval")
def main(cfg: DictConfig) -> None:
    logger.info(f"Evaluation Configuration: \n{OmegaConf.to_yaml(cfg)}")

    eval_runner = EvalRunner(config=cfg)
    try:
        eval_runner.run_evaluation()
    finally: 
        eval_runner.cleanup()


if __name__ == "__main__":
    main()
