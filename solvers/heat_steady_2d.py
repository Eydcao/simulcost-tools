import torch
import h5py
import os
from .base_solver import SIMULATOR
import numpy as np
import matplotlib.pyplot as plt
import json


class SteadyHeat2D(SIMULATOR):
    """2D steady-state heat transfer solver using point-wise Jacobi with SOR"""

    def __init__(self, verbose, cfg):
        # Physical parameters
        self.Lx = 1.0  # Domain size in x
        self.Ly = 1.0  # Domain size in y
        self.T_top = cfg.T_top  # Fixed top boundary temperature
        self.T_bottom = cfg.T_bottom  # Fixed bottom boundary temperature
        self.T_left = cfg.T_left  # Fixed left boundary temperature
        self.T_right = cfg.T_right  # Fixed right boundary temperature

        # Numerical parameters
        self.dx = cfg.dx  # Grid spacing in x
        self.nx = int(self.Lx / self.dx + 1)  # Grid points in x
        self.ny = int(self.Ly / self.dx + 1)  # Grid points in y
        self.relax = cfg.relax  # SOR relaxation parameter (1 < relax < 2)
        self.error_threshold = cfg.error_threshold  # Convergence threshold
        self.T_init = cfg.T_init  # Initial temperature field

        # Create output directory
        self.dump_dir = (
            cfg.dump_dir + f"_dx{self.dx}_relax_{self.relax}_Tinit_{self.T_init}_error_{self.error_threshold}/"
        )
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Initialize grid
        self.x = torch.linspace(0, self.Lx, self.nx)
        self.y = torch.linspace(0, self.Ly, self.ny)

        # Initialize temperature field (including boundaries)
        self.T = torch.ones((self.nx, self.ny)) * self.T_init  # Default to initial temperature

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

        # Set a clone for T old
        self.T_old = self.T.clone()
        self.converged = False

        # Base initialization
        super().__init__(verbose, cfg)

    def cal_dt(self):
        """For steady-state, we use fixed iteration steps"""
        return 1.0  # Each 'step' is one iteration

    @torch.no_grad()
    def step(self, dt):
        """Perform one Jacobi iteration with SOR"""
        self.T_old = self.T.clone()

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
        # calculate the rmse between current and previous temperature field
        # if very small, then stop
        diff = torch.sqrt(torch.mean((self.T - self.T_old) ** 2))
        if diff < self.error_threshold:
            # make a final dump
            self.dump()
            print(f"Converged with error {diff:.6f} at step {self.num_steps}, stopping simulation.")
            self.converged = True
            return True
        return False

    def dump(self):
        """Save current state including data file and visualization"""
        file_base = os.path.join(self.dump_dir, f"res_{self.num_steps}")

        # Save HDF5 data file
        with h5py.File(f"{file_base}.h5", "w") as f:
            f.create_dataset("x", data=self.x.numpy())
            f.create_dataset("y", data=self.y.numpy())
            f.create_dataset("T", data=self.T.numpy())
            f.create_dataset("iter", data=self.current_time)  # Using current_time as iteration count

        # Create and save plot
        plt.figure(figsize=(10, 8))
        X, Y = torch.meshgrid(self.x, self.y, indexing="ij")
        plt.contourf(X.numpy(), Y.numpy(), self.T.numpy(), levels=20, cmap="jet")
        plt.colorbar(label="Temperature")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Iteration = {int(self.current_time)}")
        plt.savefig(f"{file_base}.png")
        plt.close()

    def post_process(self):
        """Calculate and save computational cost"""
        cost = (self.nx * self.ny) * self.num_steps
        meta = {
            "cost": cost,
            "num_steps": self.num_steps,          # ➜ 新增
            "converged": int(self.converged)
        }
        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        print(f"Total cost: {cost}, num_steps: {self.num_steps}")