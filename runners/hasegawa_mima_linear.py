import hydra
from omegaconf import DictConfig
from solvers.hasegawa_mima_linear import HasegawaMimaLinear


@hydra.main(version_base=None, config_path="../run_configs/hasegawa_mima_linear", config_name="p1")
def main(cfg: DictConfig) -> None:
    """Main entry point for linear Hasegawa-Mima simulation"""

    # Create and run simulation
    simulator = HasegawaMimaLinear(verbose=cfg.verbose, cfg=cfg)

    # Run simulation using base framework
    simulator.run()

    if cfg.verbose:
        method_name = "Analytical" if cfg.get('analytical', False) else "Numerical"
        print(f"{method_name} simulation completed. Results saved to: {simulator.dump_dir}")


if __name__ == "__main__":
    main()