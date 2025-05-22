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
        self.w = cfg.w  # the one parameter for the minmod limiter
        self.k = cfg.k  # the blending parameter between the central (1) and upwind (-1) fluxes

        # Create spatial grid (without endpoint for periodic domain)
        self.x = np.linspace(0, self.domain_length, self.n_space, endpoint=False)

        # Initialize solution with the given initial condition
        # Default: u(x,0) = sin(2π*x/L) + 0.5
        self.u = np.sin(2 * np.pi * self.x / self.domain_length) + 0.5

        # Output directory
        self.dump_dir = cfg.dump_dir + f"_cfl_{self.cfl}_k_{self.k}_w_{self.w}"
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Base initialization
        super().__init__(verbose, cfg)

    def cal_dt(self):
        """
        Calculate timestep using CFL condition for Burgers equation.
        For Burgers equation, the wave speed is the solution itself.
        """
        max_speed = np.max(np.abs(self.u))
        # Add small epsilon to prevent division by zero
        dt = self.cfl * self.dx / (max_speed + 1e-10)
        return dt

    def minmod(self, a, b):
        """
        Vectorized minmod flux limiter function.
        Returns 0 where a and b have different signs,
        otherwise returns the smaller absolute value with the sign preserved.
        """
        # Vectorized implementation
        result = np.zeros_like(a)
        mask = a * b > 0
        result[mask] = np.sign(a[mask]) * np.minimum(np.abs(a[mask]), np.abs(b[mask]))
        return result

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
        Perform a single time step using 2nd order Roe method with minmod limiter.
        Fully vectorized implementation.
        """
        # Add ghost cells
        ug = self.get_ghost_cells(self.u)
        N = len(self.u)

        # Vectorized slope calculation using minmod limiter
        left_diff = ug[2:-2] - ug[1:-3]  # u[i] - u[i-1]
        right_diff = ug[3:-1] - ug[2:-2]  # u[i+1] - u[i]

        # Apply minmod limiter to all interior cells at once
        slopes_left = np.zeros(N + 4)
        slopes_right = np.zeros(N + 4)

        # Compute slopes with weighting factor self.w
        slopes_left[2:-2] = self.minmod(self.w * left_diff, right_diff)
        slopes_right[2:-2] = self.minmod(left_diff, self.w * right_diff)

        # Apply periodic boundary conditions to slopes
        slopes_left[1] = slopes_left[-3]
        slopes_left[0] = slopes_left[-4]
        slopes_left[-2] = slopes_left[2]
        slopes_left[-1] = slopes_left[3]

        slopes_right[1] = slopes_right[-3]
        slopes_right[0] = slopes_right[-4]
        slopes_right[-2] = slopes_right[2]
        slopes_right[-1] = slopes_right[3]

        # Reconstruct left and right states in vectorized form
        # Indexing explained: u_left[i] corresponds to left state at i-1/2 interface
        u_left = ug[1:-3] + 0.25 * (1 + self.k) * slopes_left[1:-3] + 0.25 * (1 - self.k) * slopes_right[0:-4]
        u_right = ug[2:-2] - 0.25 * (1 + self.k) * slopes_right[2:-2] - 0.25 * (1 - self.k) * slopes_left[3:-1]

        # u_left = ug[1:-3] + 0.5 * slopes_left[1:-3]
        # u_right = ug[2:-2] - 0.5 * slopes_right[2:-2]

        # u_left = ug[1:-3] + 0.5 * slopes_right[0:-4]
        # u_right = ug[2:-2] - 0.5 * slopes_left[3:-1]

        # Compute Roe fluxes at all interfaces simultaneously
        f_left = 0.5 * u_left**2  # Flux function for Burgers: f(u) = 0.5*u^2
        f_right = 0.5 * u_right**2

        # Roe-averaged wave speeds
        a = 0.5 * (u_left + u_right)

        # Vectorized Roe flux formula
        F = 0.5 * (f_left + f_right) - 0.5 * np.abs(a) * (u_right - u_left)

        # Boundary flux for periodic conditions
        F = np.append(F, F[0])

        # Update solution in vectorized form
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
                "total_steps": int(self.num_steps),
            }
            json.dump(meta, f, indent=4)
        print(f"Total cost: {cost}")
