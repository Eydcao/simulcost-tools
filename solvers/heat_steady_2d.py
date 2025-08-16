import h5py
import os
from .base_solver import SIMULATOR
from .utils import format_param_for_path
import numpy as np
import matplotlib.pyplot as plt
import json


class SteadyHeat2D(SIMULATOR):
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

        # Derived parameters
        self.nx = int(self.Lx / self.dx + 1)  # Grid points in x
        self.ny = int(self.Ly / self.dx + 1)  # Grid points in y

        # Create output directory with clean formatting
        self.dump_dir = (
            cfg.dump_dir
            + f"_dx_{format_param_for_path(self.dx)}_relax_{format_param_for_path(self.relax)}_Tinit_{format_param_for_path(self.T_init)}_error_{format_param_for_path(self.error_threshold)}"
        )
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Initialize grid
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)

        # Initialize temperature field (including boundaries)
        self.T = np.ones((self.nx, self.ny)) * self.T_init

        # Set boundary conditions
        self._apply_boundary_conditions()

        # Set a clone for T old
        self.T_old = self.T.copy()
        self.converged = False
        self.numerical_instability = False

        # Base initialization
        super().__init__(verbose, cfg)

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

    def cal_dt(self):
        """For steady-state, we use fixed iteration steps"""
        return 1.0  # Each 'step' is one iteration

    def step(self, dt):
        """Perform one Jacobi iteration with SOR"""
        self.T_old = self.T.copy()

        T_c = self.T_old[1:-1, 1:-1]  # Center points
        T_l = self.T_old[:-2, 1:-1]  # Left points
        T_r = self.T_old[2:, 1:-1]  # Right points
        T_b = self.T_old[1:-1, :-2]  # Bottom points
        T_t = self.T_old[1:-1, 2:]  # Top points

        T_iter = 0.25 * (T_l + T_r + T_b + T_t)  # Jacobi update
        T_new = self.relax * T_iter + (1.0 - self.relax) * T_c  # SOR relaxation

        # Update the field for interior points only
        self.T[1:-1, 1:-1] = T_new

    def early_stop(self):
        # Check for numerical instability (NaN or infinite values)
        if not np.all(np.isfinite(self.T)):
            # Record the final frame with NaN values for failure analysis
            self.numerical_instability = True
            self.dump()
            print(f"Numerical instability detected (NaN/Inf values) at step {self.num_steps}, stopping simulation.")
            self.converged = False
            return True

        # calculate the rmse between current and previous temperature field
        # if very small, then stop
        diff = np.sqrt(np.mean((self.T - self.T_old) ** 2))
        if diff < self.error_threshold:
            # make a final dump
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
            }
            json.dump(meta, f, indent=4)
        if self.verbose:
            print(f"Run cost: {cost}")
