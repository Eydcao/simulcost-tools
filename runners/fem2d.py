import sys
import os

# Add repository root to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

# Add fastipc_utils to Python path for internal imports
fastipc_utils_path = os.path.join(repo_root, "solvers", "fastipc_utils")
sys.path.append(fastipc_utils_path)

import hydra
from omegaconf import DictConfig, OmegaConf
from solvers.fem2d import FEM2D
from solvers.utils import format_param_for_path


@hydra.main(version_base=None, config_path="../run_configs/fem2d", config_name="p1")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Override dump_dir to include parameters in the directory name
    # Extract profile name from dump_dir (e.g., "sim_res/fem2d/p1" -> "p1")
    profile = os.path.basename(cfg.dump_dir)
    base_dir = os.path.dirname(cfg.dump_dir)

    # Create parameterized directory name
    param_dir = (
        f"{profile}_nx{cfg.nx}_dt{format_param_for_path(cfg.dt)}_nvrestol{format_param_for_path(cfg.newton_v_res_tol)}"
    )
    cfg.dump_dir = os.path.join(base_dir, param_dir)

    solver = FEM2D(verbose=cfg.verbose, cfg=cfg)
    solver.run()


if __name__ == "__main__":
    main()
