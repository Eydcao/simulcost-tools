import hydra
from omegaconf import OmegaConf
from solvers import *
import os

os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"


@hydra.main(version_base=None, config_path="run_configs/heat_1d", config_name="p1")
def main(cfg):
    # Print config (for debugging)
    print(OmegaConf.to_yaml(cfg))

    # Initialize and run solver
    solver = Heat1D(verbose=cfg.verbose, cfg=cfg)
    solver.run()

    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()
