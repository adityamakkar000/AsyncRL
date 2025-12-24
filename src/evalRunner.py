
import hydra
from src.eval.eval import EvalRunner
from src.eval.config import (
    vLLMConfig, 
    modelConfig,
    evalConfig, 
)
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, MISSING

cs = ConfigStore.instance()
cs.store(name="base_eval_config", node=evalConfig)

@hydra.main(version_base=None, config_path="./configs/eval")
def main(cfg: DictConfig) -> None:
    print("Evaluation Configuration:")
    print(OmegaConf.to_yaml(cfg))
    breakpoint()


if __name__ == "__main__":
    main()