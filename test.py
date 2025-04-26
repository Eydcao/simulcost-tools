import hydra
from omegaconf import OmegaConf
from heat_1d import Heat1D


@hydra.main(version_base=None, config_path="run_configs", config_name="heat_1d")
def main(cfg):
    # Print config (for debugging)
    print(OmegaConf.to_yaml(cfg))

    # Initialize and run solver
    solver = Heat1D(verbose=cfg.verbose, cfg=cfg)
    solver.run()

    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()
