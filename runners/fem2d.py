import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import hydra
from omegaconf import DictConfig, OmegaConf
from solvers.fem2d import FEM2D

@hydra.main(version_base=None, config_path="../run_configs/fem2d", config_name="p1")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    solver = FEM2D(verbose=cfg.verbose, cfg=cfg)
    solver.run()

if __name__ == "__main__":
    main()
