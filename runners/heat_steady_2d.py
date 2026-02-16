import hydra
from omegaconf import OmegaConf

# Append abs path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.heat_steady_2d import SteadyHeat2D


@hydra.main(version_base=None, config_path="../run_configs/heat_steady_2d", config_name="p1")
def main(cfg):
    # Print config (for debugging)
    print(OmegaConf.to_yaml(cfg))

    # Initialize and run solver
    solver = SteadyHeat2D(verbose=cfg.verbose, cfg=cfg)
    solver.run()

    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()
