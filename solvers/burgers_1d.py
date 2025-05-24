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
        self.case = cfg.case
        self.u = self.initialize_condition(self.case)

        # Output directory
        self.dump_dir = cfg.dump_dir + f"_cfl_{self.cfl}_k_{self.k}_w_{self.w}"
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
        slopes_left[2:-2] = self.minmod(self.w * left_diff, right_diff)  # j+1/2, -
        slopes_right[2:-2] = self.minmod(left_diff, self.w * right_diff)  # j-1/2, +

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
        u_left = ug[1:-3] + 0.25 * (1 - self.k) * slopes_right[1:-3] + 0.25 * (1 + self.k) * slopes_left[1:-3]
        u_right = ug[2:-2] - 0.25 * (1 + self.k) * slopes_right[2:-2] - 0.25 * (1 - self.k) * slopes_left[2:-2]

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

    def find_wave_regions(self, u):
        """
        Detect shock and rarefaction wave regions in the solution.

        Args:
            u: 1D array of solution values

        Returns:
            shocks: List of tuples (start_idx, end_idx) for shock regions
            rarefactions: List of tuples (start_idx, end_idx) for rarefaction regions
        """
        # Parameters
        shock_threshold = 0.5  # Minimum gradient magnitude to consider a shock
        # smooth_threshold = 0.1  # Maximum gradient for smooth regions
        min_wave_width = 3  # Minimum points to consider a wave

        # Compute first and second derivatives
        du = np.gradient(u, self.dx)
        # d2u = np.gradient(du, self.dx)

        # Initialize wave regions
        shocks = []
        # rarefactions = []

        # First pass: Identify candidate regions based on gradient magnitude
        in_shock = False
        # in_rarefaction = False
        start_idx = 0

        for i in range(1, len(u) - 1):
            # Shock detection (large gradient + compression)
            if abs(du[i]) > shock_threshold:
                # Check for compression (du/dx negative for Burgers shocks)
                if du[i] < 0 and not in_shock:
                    in_shock = True
                    start_idx = i
                elif du[i] >= 0 and in_shock:
                    in_shock = False
                    if i - start_idx > min_wave_width:
                        shocks.append((start_idx, i))

            # # Rarefaction detection (smooth, monotonic)
            # elif abs(du[i]) < smooth_threshold and not in_shock:
            #     # Check for consistent sign in first and second derivatives
            #     if (du[i] * du[i + 1] > 0) and (d2u[i] * du[i] > -0.1):  # Mild curvature
            #         if not in_rarefaction:
            #             in_rarefaction = True
            #             start_idx = i
            #     elif in_rarefaction:
            #         in_rarefaction = False
            #         if i - start_idx > min_wave_width:
            #             rarefactions.append((start_idx, i))

        # # Add any final regions # TODO make cyclic
        # if in_shock and len(u) - 1 - start_idx > min_wave_width:
        #     shocks.append((start_idx, len(u) - 1))
        # if in_rarefaction and len(u) - 1 - start_idx > min_wave_width:
        #     rarefactions.append((start_idx, len(u) - 1))

        # Second pass: Refine boundaries using characteristics
        refined_shocks = []
        for start, end in shocks:
            # Find peak gradient location
            peak_idx = start + np.argmax(np.abs(du[start : end + 1]))
            # Expand to include full transition
            left = max(0, peak_idx - 2)
            right = min(len(u) - 1, peak_idx + 2)
            refined_shocks.append((left, right))

        # refined_rarefactions = []
        # for start, end in rarefactions:
        #     # Find consistent expansion region
        #     u_start = u[start]
        #     u_end = u[end]
        #     if u_end > u_start:  # Increasing rarefaction
        #         left = start
        #         while left > 0 and u[left - 1] <= u[left] and du[left] > 0:
        #             left -= 1
        #         right = end
        #         while right < len(u) - 1 and u[right + 1] >= u[right] and du[right] > 0:
        #             right += 1
        #     else:  # Decreasing rarefaction
        #         left = start
        #         while left > 0 and u[left - 1] >= u[left] and du[left] < 0:
        #             left -= 1
        #         right = end
        #         while right < len(u) - 1 and u[right + 1] <= u[right] and du[right] < 0:
        #             right += 1
        #     refined_rarefactions.append((left, right))

        # return refined_shocks, refined_rarefactions
        return refined_shocks, []

    def measure_wave_width(self, u, regions):
        """Measure width of wave regions (works for both shocks and rarefactions)"""
        widths = []
        for start, end in regions:
            u_start = u[start]
            u_end = u[end]
            transition = (u[start : end + 1] - u_end) / (u_start - u_end + 1e-10)
            left_idx = np.argmax(transition >= 0.1)
            right_idx = len(transition) - np.argmax(transition[::-1] <= 0.9) - 1
            widths.append(right_idx - left_idx)
        return widths

    def dump(self):
        """Save current state with wave detection"""
        # Detect both wave types
        shocks, rarefactions = self.find_wave_regions(self.u)
        shock_widths = self.measure_wave_width(self.u, shocks)
        rare_widths = self.measure_wave_width(self.u, rarefactions)

        # Prepare JSON data with type conversion
        wave_data = {
            "shocks": {"regions": [(int(s[0]), int(s[1])) for s in shocks], "widths": [int(w) for w in shock_widths]},
            "rarefactions": {
                "regions": [(int(r[0]), int(r[1])) for r in rarefactions],
                "widths": [int(w) for w in rare_widths],
            },
            "time": float(self.current_time),
        }

        # Save to JSON
        file_base = os.path.join(self.dump_dir, f"res_{self.record_frame}")
        with open(f"{file_base}_waves.json", "w") as f:
            json.dump(wave_data, f, indent=2)

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(self.x, self.u, "b-", linewidth=2, label="Solution")

        # Highlight shocks (red) and rarefactions (blue)
        for (start, end), width in zip(shocks, shock_widths):
            plt.axvspan(self.x[start], self.x[end], color="red", alpha=0.2)
            mid_x = (self.x[start] + self.x[end]) / 2
            plt.text(mid_x, np.max(self.u) * 0.9, f"Shock\n{width}c", ha="center", va="top", color="red")

        for (start, end), width in zip(rarefactions, rare_widths):
            plt.axvspan(self.x[start], self.x[end], color="blue", alpha=0.2)
            mid_x = (self.x[start] + self.x[end]) / 2
            plt.text(mid_x, np.min(self.u) * 0.9, f"Rarefaction\n{width}c", ha="center", va="bottom", color="blue")

        plt.xlabel("Position (x)")
        plt.ylabel("Solution (u)")
        plt.title(f"Burgers Equation (t={self.current_time:.3f})")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{file_base}.png")
        plt.close()

        # Original HDF5 output
        with h5py.File(f"{file_base}.h5", "w") as f:
            f.create_dataset("x", data=self.x)
            f.create_dataset("u", data=self.u)
            f.create_dataset("time", data=self.current_time)

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
        print(f"Run cost: {cost}")
