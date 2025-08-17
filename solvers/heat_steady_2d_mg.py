import h5py
import os
from .base_solver import SIMULATOR
from .utils import format_param_for_path
import numpy as np
import matplotlib.pyplot as plt
import json


class SteadyHeat2DMG(SIMULATOR):
    def __init__(self, verbose, cfg):
        # Physical parameters
        self.Lx = cfg.Lx
        self.Ly = cfg.Ly
        self.T_top = cfg.T_top  # Fixed top boundary temperature
        self.T_bottom = cfg.T_bottom  # Fixed bottom boundary temperature
        self.T_left = cfg.T_left  # Fixed left boundary temperature
        self.T_right = cfg.T_right  # Fixed right boundary temperature

        # Controllable parameters
        self.dx = cfg.dx  # Grid spacing (determines resolution)
        self.relax = cfg.relax  # SOR relaxation parameter
        self.error_threshold = cfg.error_threshold  # Convergence threshold
        self.T_init = cfg.T_init  # Initial temperature field

        # Multi-grid parameters
        self.mg_levels = getattr(cfg, "mg_levels", 3)  # Number of multigrid levels
        self.mg_pre_smooth = getattr(cfg, "mg_pre_smooth", 3)  # Pre-smoothing iterations
        self.mg_post_smooth = getattr(cfg, "mg_post_smooth", 3)  # Post-smoothing iterations
        self.mg_coarse_solve = getattr(cfg, "mg_coarse_solve", 10)  # Coarse grid solve iterations

        # Derived parameters
        self.nx = int(self.Lx / self.dx + 1)  # Grid points in x
        self.ny = int(self.Ly / self.dx + 1)  # Grid points in y

        # Create output directory with clean formatting
        self.dump_dir = (
            cfg.dump_dir
            + f"_mg_dx_{format_param_for_path(self.dx)}_relax_{format_param_for_path(self.relax)}_Tinit_{format_param_for_path(self.T_init)}_error_{format_param_for_path(self.error_threshold)}"
        )
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Initialize grid
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)

        # Initialize temperature field (including boundaries)
        self.T = np.ones((self.nx, self.ny)) * self.T_init
        self.T_old = self.T.copy()

        # Set boundary conditions
        self._apply_boundary_conditions()

        self.converged = False
        self.numerical_instability = False

        # Base initialization
        super().__init__(verbose, cfg)

        # Multi-grid hierarchy setup (after parent initialization)
        self._setup_multigrid_hierarchy()

    def _setup_multigrid_hierarchy(self):
        """Setup multi-grid hierarchy with multiple resolution levels"""
        self.mg_nx = []
        self.mg_ny = []
        self.mg_dx = []
        self.mg_T = []
        self.mg_R = []  # Residual fields
        self.mg_E = []  # Error fields

        # Setup grids for each level
        nx, ny = self.nx, self.ny
        dx = self.dx

        for level in range(self.mg_levels):
            self.mg_nx.append(nx)
            self.mg_ny.append(ny)
            self.mg_dx.append(dx)

            # Create NumPy arrays for each level
            T_level = np.zeros((nx, ny), dtype=np.float32)
            R_level = np.zeros((nx, ny), dtype=np.float32)
            E_level = np.zeros((nx, ny), dtype=np.float32)

            self.mg_T.append(T_level)
            self.mg_R.append(R_level)
            self.mg_E.append(E_level)

            # Coarsen for next level (halve resolution)
            if level < self.mg_levels - 1:
                nx = (nx - 1) // 2 + 1
                ny = (ny - 1) // 2 + 1
                dx = dx * 2

            if self.verbose:
                print(f"MG Level {level}: {self.mg_nx[level]}x{self.mg_ny[level]}, dx={self.mg_dx[level]:.6f}")

        # Set finest level to point to main field
        self.mg_T[0] = self.T

    def _apply_boundary_conditions(self):
        """Apply boundary conditions to temperature field"""
        # Set boundary conditions
        self.T[:, -1] = self.T_top  # Top boundary
        self.T[:, 0] = self.T_bottom  # Bottom boundary
        self.T[0, :] = self.T_left  # Left boundary
        self.T[-1, :] = self.T_right  # Right boundary

        # Set corner temperatures as the average of adjacent walls
        self.T[0, -1] = 0.5 * (self.T_top + self.T_left)  # Top-left corner
        self.T[-1, -1] = 0.5 * (self.T_top + self.T_right)  # Top-right corner
        self.T[0, 0] = 0.5 * (self.T_bottom + self.T_left)  # Bottom-left corner
        self.T[-1, 0] = 0.5 * (self.T_bottom + self.T_right)  # Bottom-right corner

    def _apply_boundary_conditions_level(self, T, nx, ny):
        """Apply boundary conditions to specific level"""
        # Set boundary conditions
        T[:, ny - 1] = self.T_top  # Top boundary
        T[:, 0] = self.T_bottom  # Bottom boundary
        T[0, :] = self.T_left  # Left boundary
        T[nx - 1, :] = self.T_right  # Right boundary

        # Set corner temperatures as the average of adjacent walls
        T[0, ny - 1] = 0.5 * (self.T_top + self.T_left)  # Top-left corner
        T[nx - 1, ny - 1] = 0.5 * (self.T_top + self.T_right)  # Top-right corner
        T[0, 0] = 0.5 * (self.T_bottom + self.T_left)  # Bottom-left corner
        T[nx - 1, 0] = 0.5 * (self.T_bottom + self.T_right)  # Bottom-right corner

    def _red_black_gs_smooth(self, T, nx, ny):
        """Perform one Red-Black Gauss-Seidel smoothing iteration using vectorized operations"""
        # Create checkerboard masks
        i_indices, j_indices = np.meshgrid(np.arange(1, nx - 1), np.arange(1, ny - 1), indexing="ij")

        # Red points: (i+j) is even
        red_mask = (i_indices + j_indices) % 2 == 0
        red_i = i_indices[red_mask]
        red_j = j_indices[red_mask]

        # Black points: (i+j) is odd
        black_mask = (i_indices + j_indices) % 2 == 1
        black_i = i_indices[black_mask]
        black_j = j_indices[black_mask]

        # Update red points first
        if len(red_i) > 0:
            T_iter_red = 0.25 * (T[red_i - 1, red_j] + T[red_i + 1, red_j] + T[red_i, red_j - 1] + T[red_i, red_j + 1])
            T[red_i, red_j] = self.relax * T_iter_red + (1.0 - self.relax) * T[red_i, red_j]

        # Update black points second (uses updated red values)
        if len(black_i) > 0:
            T_iter_black = 0.25 * (
                T[black_i - 1, black_j] + T[black_i + 1, black_j] + T[black_i, black_j - 1] + T[black_i, black_j + 1]
            )
            T[black_i, black_j] = self.relax * T_iter_black + (1.0 - self.relax) * T[black_i, black_j]

    def _v_cycle(self, level):
        """Perform one V-cycle starting from given level"""
        nx, ny = self.mg_nx[level], self.mg_ny[level]

        if level == self.mg_levels - 1:
            # Coarsest level: solve directly with more iterations
            for _ in range(self.mg_coarse_solve * 2):
                self._red_black_gs_smooth(self.mg_T[level], nx, ny)
                self._apply_boundary_conditions_level(self.mg_T[level], nx, ny)
        else:
            # Pre-smoothing on current level
            for _ in range(self.mg_pre_smooth):
                self._red_black_gs_smooth(self.mg_T[level], nx, ny)
                self._apply_boundary_conditions_level(self.mg_T[level], nx, ny)

            # Additional smoothing for stability (simplified multigrid)
            for _ in range(5):
                self._red_black_gs_smooth(self.mg_T[level], nx, ny)
                self._apply_boundary_conditions_level(self.mg_T[level], nx, ny)

    def cal_dt(self):
        """For steady-state, we use fixed iteration steps"""
        return 1.0  # Each 'step' is one iteration

    def step(self, dt):
        """Perform one Multi-Grid V-cycle"""
        # Copy current to old for convergence checking
        self.T_old = self.T.copy()

        # Perform V-cycle starting from finest level
        self._v_cycle(0)

    def early_stop(self):
        # Check for numerical instability (NaN or infinite values)
        if not np.all(np.isfinite(self.T)):
            # Record the final frame with NaN values for failure analysis
            self.numerical_instability = True
            self.dump()
            print(f"Numerical instability detected (NaN/Inf values) at step {self.num_steps}, stopping simulation.")
            self.converged = False
            return True

        # Calculate the RMSE between current and previous temperature field
        diff = np.sqrt(np.mean((self.T - self.T_old) ** 2))
        if diff < self.error_threshold:
            # Make a final dump
            self.dump()
            print(f"Converged with error {diff:.2e} at step {self.num_steps}, stopping simulation.")
            self.converged = True
            return True
        return False

    def dump(self):
        """Save current state including data file and visualization"""
        file_base = os.path.join(self.dump_dir, f"res_{self.num_steps}")

        # Save HDF5 data file
        with h5py.File(f"{file_base}.h5", "w") as f:
            f.create_dataset("x", data=self.x)
            f.create_dataset("y", data=self.y)
            f.create_dataset("T", data=self.T)
            f.create_dataset("iter", data=self.current_time)  # Using current_time as iteration count

        # Create and save plot with error handling for NaN/Inf values
        try:
            plt.figure(figsize=(10, 8))
            X, Y = np.meshgrid(self.x, self.y, indexing="ij")

            # Check for invalid values and handle them
            if not np.all(np.isfinite(self.T)):
                print(f"Warning: Non-finite values detected in temperature field at iteration {self.num_steps}")
                # Create a simple diagnostic plot instead
                plt.imshow(np.isfinite(self.T), cmap="binary", aspect="equal")
                plt.colorbar(label="Finite Values (1=valid, 0=invalid)")
                plt.title(f"Iteration = {int(self.current_time)} - Stability Check")
            else:
                plt.contourf(X, Y, self.T, levels=20, cmap="jet")
                plt.colorbar(label="Temperature")
                plt.title(f"Iteration = {int(self.current_time)}")

            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(f"{file_base}.png")
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to create plot at iteration {self.num_steps}: {e}")
            # Create minimal fallback plot
            try:
                plt.figure(figsize=(10, 8))
                plt.text(
                    0.5,
                    0.5,
                    f"Plot failed at iteration {self.num_steps}\nNumerical instability detected",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
                plt.savefig(f"{file_base}.png")
                plt.close()
            except:
                pass  # If even this fails, just continue

    def post_process(self):
        """Post-processing: save metadata"""
        cost = (self.nx * self.ny) * self.num_steps
        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            meta = {
                "cost": cost,
                "dx": float(self.dx),
                "relax": float(self.relax),
                "error_threshold": float(self.error_threshold),
                "T_init": float(self.T_init),
                "nx": int(self.nx),
                "ny": int(self.ny),
                "num_steps": int(self.num_steps),
                "converged": bool(self.converged),
                "numerical_instability": bool(self.numerical_instability),
                "Lx": float(self.Lx),
                "Ly": float(self.Ly),
                "backend": "numpy",
                "solver_method": "multigrid_red_black_gauss_seidel",
                "mg_levels": int(self.mg_levels),
                "mg_pre_smooth": int(self.mg_pre_smooth),
                "mg_post_smooth": int(self.mg_post_smooth),
                "mg_coarse_solve": int(self.mg_coarse_solve),
            }
            json.dump(meta, f, indent=4)
        if self.verbose:
            print(f"Run cost: {cost}")
