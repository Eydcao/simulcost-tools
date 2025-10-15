import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import hydra
from omegaconf import DictConfig, OmegaConf
from solvers.fastipc import FastIPC

@hydra.main(version_base=None, config_path="../run_configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    solver = FastIPC(verbose=True, cfg=cfg.solver)
    solver.run()

if __name__ == "__main__":
    main()
