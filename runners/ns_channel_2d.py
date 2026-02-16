import hydra
from omegaconf import OmegaConf

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.ns_channel_2d import NSChannel2D


@hydra.main(version_base=None, config_path="../run_configs/ns_channel_2d", config_name="p1")
def main(cfg):
    # Print config (for debugging)
    print(OmegaConf.to_yaml(cfg))

    # Initialize and run solver
    solver = NSChannel2D(
        cfg=cfg,
        verbose=cfg.verbose
    )
    status = solver.run()
    if not status:
        print("Simulation failed! See logs for details.")
        return
    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()