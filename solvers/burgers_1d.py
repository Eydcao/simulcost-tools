import torch
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from .base_solver import SIMULATOR


class BurgersRoe2(SIMULATOR):
    """
    2nd order Roe method for Burgers equation with minmod limiter.
    Uses periodic boundary conditions.
    """

    def __init__(self, verbose, cfg):
        # Physical parameters
        self.domain_length = 2 * np.pi  # Domain length (0 to 2π)

        # Numerical parameters
        self.n_space = cfg.n_space  # Number of spatial points
        self.dx = self.domain_length / self.n_space
        self.cfl = cfg.cfl  # CFL number
        self.w = cfg.w  # Limiter parameter (controls minmod limiter)
        self.k = cfg.k if hasattr(cfg, "k") else -1  # Method parameter, default -1

        # Create spatial grid (including endpoint for periodic domain)
        self.nx = self.n_space + 1
        self.x = torch.linspace(0, self.domain_length, self.nx)

        # Initialize solution with the given initial condition
        # u(x,0) = sin(x) + 0.5*sin(0.5*x)
        self.u = torch.sin(self.x) + 0.5 * torch.sin(0.5 * self.x)

        # Output directory
        self.dump_dir = cfg.dump_dir + f"_w_{self.w}_nx_{self.n_space}/"
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Base initialization
        super().__init__(verbose, cfg)

    def cal_dt(self):
        """
        Calculate timestep using CFL condition for Burgers equation.
        For Burgers equation, the wave speed is the solution itself.
        """
        max_speed = torch.max(torch.abs(self.u))
        dt = self.cfl * self.dx / max_speed
        return dt

    def minmod(self, x, y):
        """
        Minmod flux limiter function.
        Returns 0 if x and y have different signs,
        otherwise returns the smaller absolute value with the sign preserved.
        """
        result = torch.zeros_like(x)
        valid_mask = x * y > 0
        result[valid_mask] = torch.sign(x[valid_mask]) * torch.min(torch.abs(x[valid_mask]), torch.abs(y[valid_mask]))
        return result

    def periodic_index(self, i):
        """Helper function to handle periodic boundary conditions"""
        return i % self.nx

    @torch.no_grad()
    def step(self, dt):
        """
        Perform a single time step using 2nd order Roe method with minmod limiter.
        """
        up = self.u.clone()
        f = torch.zeros_like(up)
        ul = torch.zeros_like(up)
        ur = torch.zeros_like(up)
        dp = torch.zeros_like(up)
        dn = torch.zeros_like(up)

        # Calculate slopes with minmod limiter
        for i in range(self.nx - 1):
            dn[i] = self.minmod(up[i + 1] - up[i], self.w * (up[i] - up[self.periodic_index(i - 1)]))
            dp[i] = self.minmod(up[i + 1] - up[i], self.w * (up[self.periodic_index(i + 2)] - up[i + 1]))

        # Handle last point for periodic boundary
        dn[-1] = dn[0]
        dp[-1] = dp[0]

        # Reconstruct left and right states
        for i in range(self.nx - 1):
            ul[i] = up[i] + (1 - self.k) / 4 * dp[self.periodic_index(i - 1)] + (1 + self.k) / 4 * dn[i]
            ur[i] = up[i + 1] - (1 - self.k) / 4 * dn[i + 1] - (1 + self.k) / 4 * dp[i]

        # Handle last point for periodic boundary
        ul[-1] = ul[0]
        ur[-1] = ur[0]

        # Calculate average and absolute wave speed
        ubar = (ur + ul) / 2
        ubar = torch.maximum(ubar, torch.maximum(torch.zeros_like(ubar), (ur - ul) / 2))

        # Compute numerical flux
        for i in range(self.nx - 1):
            f[i] = (ul[i] ** 2 + ur[i] ** 2) / 4 - 0.5 * abs(ubar[i]) * (ur[self.periodic_index(i + 1)] - ul[i])

        # Handle last point for periodic boundary
        f[-1] = f[0]

        # Update solution
        for i in range(self.nx - 1):
            self.u[i] = up[i] - dt / self.dx * (f[i] - f[self.periodic_index(i - 1)])

        # Enforce periodic boundary condition
        self.u[-1] = self.u[0]

    def dump(self):
        """Save current state including data file and visualization"""
        # Create filename base
        file_base = os.path.join(self.dump_dir, f"res_{self.record_frame}")

        # Save HDF5 data file
        with h5py.File(f"{file_base}.h5", "w") as f:
            f.create_dataset("x", data=self.x.numpy())
            f.create_dataset("u", data=self.u.numpy())
            f.create_dataset("time", data=self.current_time)

        # Create and save plot
        plt.figure(figsize=(8, 5))
        plt.plot(self.x.numpy(), self.u.numpy(), "b-", linewidth=2)
        plt.xlabel("Position (x)")
        plt.ylabel("Solution (u)")
        plt.title(f"Time = {self.current_time:.3f}")
        plt.grid(True)
        plt.savefig(f"{file_base}.png")
        plt.close()

    def post_process(self):
        # Save the cost estimation in a json file in dump dir
        cost = self.num_steps * self.nx
        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            meta = {"cost": cost, "w_parameter": float(self.w), "k_parameter": float(self.k)}
            json.dump(meta, f, indent=4)
        print(f"Total cost: {cost}")
