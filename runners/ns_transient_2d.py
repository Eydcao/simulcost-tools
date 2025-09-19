import hydra
from omegaconf import OmegaConf

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers import *


@hydra.main(version_base=None, config_path="../run_configs/ns_transient_2d", config_name="p1")
def main(cfg):
    # Print config (for debugging)
    print(OmegaConf.to_yaml(cfg))

    # Initialize and run solver
    solver = NSTransient2D(args=cfg)
    try:
        solver.run()
        print("Simulation completed successfully!")
    except Exception as e:
        print(f"Simulation failed! Error: {e}")
        return


if __name__ == "__main__":
    main()
