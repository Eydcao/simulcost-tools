import hydra
from omegaconf import OmegaConf

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.hasegawa_mima_nonlinear import HasegawaMimaNonlinear


@hydra.main(version_base=None, config_path="../run_configs/hasegawa_mima_nonlinear", config_name="p1")
def main(cfg) -> None:
    """Main entry point for nonlinear Hasegawa-Mima simulation"""

    # Create and run simulation
    simulator = HasegawaMimaNonlinear(verbose=cfg.verbose, cfg=cfg)

    # Run simulation using base framework
    simulator.run()

    if cfg.verbose:
        print(f"Nonlinear simulation completed. Results saved to: {simulator.dump_dir}")


if __name__ == "__main__":
    main()
