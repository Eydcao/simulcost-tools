import hydra
from omegaconf import DictConfig

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.compaction import Compaction


@hydra.main(version_base=None, config_path="../run_configs/compaction", config_name="p1")
def main(cfg: DictConfig) -> None:
    """Main entry point for compaction simulation"""

    # Create and run simulation
    simulator = Compaction(verbose=cfg.verbose, cfg=cfg)

    # Pre-process
    simulator.pre_process()

    # For static problems, we just need one "frame"
    simulator.dump()

    # Call back
    simulator.call_back()

    # Post-process (save metadata including cost)
    simulator.post_process()

    if cfg.verbose:
        print(f"Simulation completed. Results saved to: {simulator.dump_dir}")


if __name__ == "__main__":
    main()