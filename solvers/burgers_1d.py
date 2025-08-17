import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import json
from .base_solver import SIMULATOR


class BurgersRoe2(SIMULATOR):
    """
    2nd order Roe method for Burgers equation with _minmod limiter.
    Uses periodic boundary conditions.
    """

    def __init__(self, verbose, cfg):
        # Physical parameters
        self.domain_length = cfg.L

        # Numerical parameters
        self.n_space = cfg.n_space  # Number of grid points
        self.dx = self.domain_length / self.n_space

        # controllable parameters
        self.cfl = cfg.cfl  # CFL number
        self.beta = cfg.beta  # Limiter parameter for generalized superbee (was w)
        self.k = cfg.k  # the blending parameter between the central (1) and upwind (-1) fluxes

        # Create spatial grid (without endpoint for periodic domain)
        self.x = np.linspace(0, self.domain_length, self.n_space, endpoint=False)

        # Initialize solution with the given initial condition
        self.case = cfg.case
        self.u = self.initialize_condition(self.case)

        # Output directory
        self.dump_dir = cfg.dump_dir + f"_cfl_{self.cfl}_k_{self.k}_beta_{self.beta}_n_{self.n_space}"
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Base initialization
        super().__init__(verbose, cfg)

    def initialize_condition(self, case):
        """Initialize with various common initial conditions for Burgers equation"""
        if case == "sin":
            # Default: u(x,0) = sin(2π*x/L) + 0.5
            return np.sin(2 * np.pi * self.x / self.domain_length) + 0.5

        elif case == "rarefaction":
            # Moving rarefaction wave
            u = np.ones_like(self.x)
            middle = self.domain_length / 2
            u[self.x < middle] = -0.1  # Lower value on left
            u[self.x >= middle] = 0.5  # Higher value on right
            return u

        elif case == "sod":
            # Sod shock tube problem modified for Burgers equation
            u = np.ones_like(self.x)
            middle = self.domain_length / 2
            u[self.x < middle] = 1.0
            u[self.x >= middle] = 0.1
            return u

        elif case == "double_shock":
            # Two interacting shock waves
            u = np.ones_like(self.x) * 0.5
            left_third = self.domain_length / 3
            right_third = 2 * self.domain_length / 3
            u[self.x < left_third] = 1.0
            u[(self.x >= left_third) & (self.x < right_third)] = 0.5
            u[self.x >= right_third] = 0.1
            return u

        elif case == "blast":
            # Interacting blast waves
            u = np.zeros_like(self.x)
            # Create two blast wave centers
            center1 = self.domain_length * 0.25
            center2 = self.domain_length * 0.75
            # Set blast wave profiles (Gaussian)
            sigma = self.domain_length / 20
            u += 1.0 * np.exp(-((self.x - center1) ** 2) / (2 * sigma**2))
            u += 0.8 * np.exp(-((self.x - center2) ** 2) / (2 * sigma**2))
            return u

        else:
            print(f"Warning: Unknown case '{case}'. Using default sine wave.")
            return np.sin(2 * np.pi * self.x / self.domain_length) + 0.5

    def cal_dt(self):
        """
        Calculate timestep using CFL condition for Burgers equation.
        For Burgers equation, the wave speed is the solution itself.
        """
        max_speed = np.max(np.abs(self.u))
        # Add small epsilon to prevent division by zero
        dt = self.cfl * self.dx / (max_speed + 1e-10)
        return dt

    def _slope_limiter(self, r):
        """Generalized superbee limiter function for MUSCL scheme (from Euler solver)"""
        psi_r = np.maximum(0, np.maximum(np.minimum(self.beta * r, 1.0), np.minimum(r, self.beta)))
        return psi_r

    def get_ghost_cells(self, u):
        """Add ghost cells with periodic boundary conditions"""
        N = len(u)
        ug = np.zeros(N + 4)
        ug[2:-2] = u
        ug[0] = u[-2]
        ug[1] = u[-1]
        ug[-2] = u[0]
        ug[-1] = u[1]
        return ug

    def step(self, dt):
        """
        Perform a single time step using 2nd order Roe method with generalized superbee limiter.
        Updated to use Euler's MUSCL reconstruction approach.
        """
        # Add ghost cells
        ug = self.get_ghost_cells(self.u)
        N = len(self.u)

        # Compute slopes for all cells using Euler's approach
        slopes = np.zeros(N + 4)

        # Vectorized slope computation for all cells (including boundaries)
        diff_left = ug[2:-2] - ug[1:-3]  # u[i] - u[i-1]
        diff_right = ug[3:-1] - ug[2:-2]  # u[i+1] - u[i]

        # Handle small denominators with vectorized conditionals (from Euler)
        small_num = np.abs(diff_left) < 1e-8
        small_den = np.abs(diff_right) < 1e-8

        # Apply conditional logic vectorized
        diff_left = np.where(small_num, 0.0, diff_left)
        diff_right = np.where(small_num, 1.0, diff_right)
        diff_left = np.where(~small_num & (diff_left > 1e-8) & small_den, 1.0, diff_left)
        diff_right = np.where(~small_num & (diff_left > 1e-8) & small_den, 1.0, diff_right)
        diff_left = np.where(~small_num & (diff_left < -1e-8) & small_den, -1.0, diff_left)
        diff_right = np.where(~small_num & (diff_left < -1e-8) & small_den, 1.0, diff_right)

        slopes[2:-2] = self._slope_limiter(diff_left / diff_right)

        # Apply periodic boundary conditions to slopes
        slopes[1] = slopes[-3]
        slopes[0] = slopes[-4]
        slopes[-2] = slopes[2]
        slopes[-1] = slopes[3]

        # Vectorized Left and Right extrapolated u-values at j+1/2 (Euler's approach)
        u_left = np.zeros(N + 1)
        u_right = np.zeros(N + 1)

        # Reconstruction for all interfaces - fully vectorized (adapted from Euler)
        # Interface i+1/2 connects cells i and i+1 (in original indexing)
        # In ghost cell indexing: cells i+2 and i+3
        indices = np.arange(N + 1)
        u_left[indices] = ug[indices + 1] + 0.25 * (1 + self.k) * slopes[indices + 1] * (
            ug[indices + 2] - ug[indices + 1]
        )
        u_right[indices] = ug[indices + 2] - 0.25 * (1 + self.k) * slopes[indices + 2] * (
            ug[indices + 3] - ug[indices + 2]
        )

        # Compute Roe fluxes at all interfaces simultaneously
        f_left = 0.5 * u_left**2  # Flux function for Burgers: f(u) = 0.5*u^2
        f_right = 0.5 * u_right**2

        # Roe-averaged wave speeds
        a = 0.5 * (u_left + u_right)

        # Vectorized Roe flux formula
        F = 0.5 * (f_left + f_right) - 0.5 * np.abs(a) * (u_right - u_left)

        # Update solution in vectorized form (periodic boundary handling)
        self.u = self.u - (dt / self.dx) * (F[1:] - F[:-1])

    def dump(self):
        """Save current state including data file and visualization"""
        # Create filename base
        file_base = os.path.join(self.dump_dir, f"res_{self.record_frame}")

        # Save HDF5 data file
        with h5py.File(f"{file_base}.h5", "w") as f:
            f.create_dataset("x", data=self.x)
            f.create_dataset("u", data=self.u)
            f.create_dataset("time", data=self.current_time)

        # Create and save plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.u, "b-", linewidth=2)
        plt.xlabel("Position (x)")
        plt.ylabel("Solution (u)")
        plt.title(f"Burgers Equation - Time = {self.current_time:.3f}")
        plt.grid(True)

        # # Add shockwave formation annotation if appropriate
        # if np.min(np.diff(self.u)) < -0.5:  # Simple heuristic to detect shocks
        #     plt.text(
        #         0.5, 0.1, "Shock wave forming", transform=plt.gca().transAxes, ha="center", fontsize=14, color="red"
        #     )

        plt.savefig(f"{file_base}.png")
        plt.close()

    def post_process(self):
        # Save the cost estimation in a json file in dump dir
        cost = self.num_steps * len(self.u)
        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            meta = {
                "cost": cost,
                "cfl": float(self.cfl),
                "beta": float(self.beta),
                "k": float(self.k),
                "n_space": int(self.n_space),
                "dx": float(self.dx),
                "total_steps": int(self.num_steps),
            }
            json.dump(meta, f, indent=4)
        print(f"Run cost: {cost}")
