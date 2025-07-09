import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import json
from .base_solver import SIMULATOR


class Euler1D(SIMULATOR):
    """
    1D Euler equations solver using MUSCL scheme with Roe flux.
    Solves the Sod shock tube problem and other 1D compressible flow problems.
    """

    def __init__(self, verbose, cfg):
        # Physical parameters
        self.L = cfg.L
        self.gamma = cfg.gamma  # Ratio of specific heats

        # Numerical parameters
        self.n_space = cfg.n_space  # Number of cells
        self.dx = self.L / self.n_space
        self.nx = self.n_space + 1  # Number of points

        # Controllable parameters
        self.cfl = cfg.cfl  # CFL number
        self.beta = cfg.beta  # Limiter parameter for generalized superbee
        self.k = cfg.k  # Blending parameter between central (k=1) and upwind (k=-1)

        # Create spatial grid
        self.x = np.linspace(self.dx / 2.0, self.L, self.nx)

        # Initialize solution with the given initial condition
        self.case = cfg.case
        self.q = self.initialize_condition(self.case)  # Conservative variables [rho, rho*u, rho*E]

        # Output directory
        self.dump_dir = cfg.dump_dir + f"_cfl_{self.cfl}_beta_{self.beta}_k_{self.k}"
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Base initialization
        super().__init__(verbose, cfg)

    def initialize_condition(self, case):
        """Initialize with various initial conditions for Euler equations"""
        r0 = np.zeros(self.nx)
        u0 = np.zeros(self.nx)
        p0 = np.zeros(self.nx)
        halfcells = int(self.n_space / 2)

        if case == "sod":
            # Sod's shock tube problem
            p0[:halfcells] = 1.0
            p0[halfcells:] = 0.1
            u0[:halfcells] = 0.0
            u0[halfcells:] = 0.0
            r0[:halfcells] = 1.0
            r0[halfcells:] = 0.125

        elif case == "lax":
            # Lax problem
            p0[:halfcells] = 3.528
            p0[halfcells:] = 0.571
            u0[:halfcells] = 0.698
            u0[halfcells:] = 0.0
            r0[:halfcells] = 0.445
            r0[halfcells:] = 0.5

        elif case == "123":
            # Test problem 123
            p0[:halfcells] = 1000.0
            p0[halfcells:] = 0.01
            u0[:halfcells] = 0.0
            u0[halfcells:] = 0.0
            r0[:halfcells] = 1.0
            r0[halfcells:] = 1.0

        else:
            print(f"Warning: Unknown case '{case}'. Using default Sod problem.")
            p0[:halfcells] = 1.0
            p0[halfcells:] = 0.1
            u0[:halfcells] = 0.0
            u0[halfcells:] = 0.0
            r0[:halfcells] = 1.0
            r0[halfcells:] = 0.125

        # Convert to conservative variables
        q = self._prim2cons(r0, u0, p0)
        return q

    def _cons2prim(self, q):
        """Convert conservative to primitive variables"""
        r = q[0]
        u = q[1] / r
        E = q[2] / r
        p = (self.gamma - 1.0) * r * (E - 0.5 * u**2)
        return (r, u, p)

    def _prim2cons(self, r, u, p):
        """Convert primitive to conservative variables"""
        E = p / ((self.gamma - 1.0) * r) + 0.5 * u**2
        q = np.array([r, r * u, r * E])
        return q

    def _prim2flux(self, q):
        """Compute flux vector for Euler equations"""
        r, u, p = self._cons2prim(q)

        F0 = r * u
        F1 = r * u**2 + p
        F2 = u * (q[2] + p)
        flux = np.array([F0, F1, F2])
        return flux

    def _slope_limiter(self, r):
        """Generalized superbee limiter function for MUSCL scheme"""
        psi_r = np.maximum(0, np.maximum(np.minimum(self.beta * r, 1.0), np.minimum(r, self.beta)))
        return psi_r

    def _flux_roe(self, q_left, q_right):
        """Vectorized Roe flux computation for all interfaces simultaneously"""
        # Compute left states (all interfaces)
        rho_L, u_L, p_L = self._cons2prim(q_left)
        h_L = self.gamma / (self.gamma - 1) * p_L / rho_L + 0.5 * u_L**2

        # Compute right states (all interfaces)
        rho_R, u_R, p_R = self._cons2prim(q_right)
        h_R = self.gamma / (self.gamma - 1) * p_R / rho_R + 0.5 * u_R**2

        # Compute Roe averages (vectorized)
        R = np.sqrt(rho_R / rho_L)
        u_ave = (R * u_R + u_L) / (R + 1)
        h_ave = (R * h_R + h_L) / (R + 1)
        a_ave = np.sqrt((self.gamma - 1.0) * (h_ave - 0.5 * u_ave * u_ave))

        # Auxiliary variables for full Roe flux (vectorized)
        alph1 = (self.gamma - 1) * u_ave * u_ave / (2 * a_ave * a_ave)
        alph2 = (self.gamma - 1) / (a_ave * a_ave)

        # Compute vector difference (vectorized)
        dw = q_right - q_left

        # Vectorized P^{-1} matrix multiplication with dw
        # P^{-1} * dw for each interface simultaneously
        Pinv_dw = np.zeros_like(dw)
        Pinv_dw[0] = (
            0.5 * (alph1 + u_ave / a_ave) * dw[0] + -0.5 * (alph2 * u_ave + 1 / a_ave) * dw[1] + alph2 / 2 * dw[2]
        )
        Pinv_dw[1] = (1 - alph1) * dw[0] + alph2 * u_ave * dw[1] + -alph2 * dw[2]
        Pinv_dw[2] = (
            0.5 * (alph1 - u_ave / a_ave) * dw[0] + -0.5 * (alph2 * u_ave - 1 / a_ave) * dw[1] + alph2 / 2 * dw[2]
        )

        # Vectorized Lambda * P^{-1} * dw (apply eigenvalues)
        eigenval_1 = np.abs(u_ave - a_ave)
        eigenval_2 = np.abs(u_ave)
        eigenval_3 = np.abs(u_ave + a_ave)

        Lambda_Pinv_dw = np.array([eigenval_1 * Pinv_dw[0], eigenval_2 * Pinv_dw[1], eigenval_3 * Pinv_dw[2]])

        # Vectorized P * Lambda * P^{-1} * dw (apply right eigenvectors)
        A_dw = np.zeros_like(dw)
        A_dw[0] = Lambda_Pinv_dw[0] + Lambda_Pinv_dw[1] + Lambda_Pinv_dw[2]
        A_dw[1] = (u_ave - a_ave) * Lambda_Pinv_dw[0] + u_ave * Lambda_Pinv_dw[1] + (u_ave + a_ave) * Lambda_Pinv_dw[2]
        A_dw[2] = (
            (h_ave - a_ave * u_ave) * Lambda_Pinv_dw[0]
            + 0.5 * u_ave * u_ave * Lambda_Pinv_dw[1]
            + (h_ave + a_ave * u_ave) * Lambda_Pinv_dw[2]
        )

        # Compute final flux (vectorized)
        flux_L = self._prim2flux(q_left)
        flux_R = self._prim2flux(q_right)

        return 0.5 * (flux_L + flux_R) - 0.5 * A_dw

    def _flux_muscl(self):
        """MUSCL reconstruction with Roe flux"""
        # Compute and limit slopes
        slope_left = np.zeros((3, self.nx - 1))
        slope_right = np.zeros((3, self.nx - 1))
        q_left = np.zeros((3, self.nx - 1))
        q_right = np.zeros((3, self.nx - 1))

        # Vectorized slope computation for all variables and cells
        # Left reconstruction slopes (for j = 1 to nx-3)
        diff_left = self.q[:, 1:-3] - self.q[:, 0:-4]  # q[i,j] - q[i,j-1]
        diff_right = self.q[:, 2:-2] - self.q[:, 1:-3]  # q[i,j+1] - q[i,j]

        # Handle small denominators with vectorized conditionals
        small_num = np.abs(diff_left) < 1e-8
        small_den = np.abs(diff_right) < 1e-8

        # Apply conditional logic vectorized
        diff_left = np.where(small_num, 0.0, diff_left)
        diff_right = np.where(small_num, 1.0, diff_right)
        diff_left = np.where(~small_num & (diff_left > 1e-8) & small_den, 1.0, diff_left)
        diff_right = np.where(~small_num & (diff_left > 1e-8) & small_den, 1.0, diff_right)
        diff_left = np.where(~small_num & (diff_left < -1e-8) & small_den, -1.0, diff_left)
        diff_right = np.where(~small_num & (diff_left < -1e-8) & small_den, 1.0, diff_right)

        slope_left[:, 1:-2] = self._slope_limiter(diff_left / diff_right)

        # Right reconstruction slopes (for j = 1 to nx-3)
        diff_left = self.q[:, 2:-2] - self.q[:, 1:-3]  # q[i,j+1] - q[i,j]
        diff_right = self.q[:, 3:-1] - self.q[:, 2:-2]  # q[i,j+2] - q[i,j+1]

        # Handle small denominators with vectorized conditionals
        small_num = np.abs(diff_left) < 1e-8
        small_den = np.abs(diff_right) < 1e-8

        # Apply conditional logic vectorized
        diff_left = np.where(small_num, 0.0, diff_left)
        diff_right = np.where(small_num, 1.0, diff_right)
        diff_left = np.where(~small_num & (diff_left > 1e-8) & small_den, 1.0, diff_left)
        diff_right = np.where(~small_num & (diff_left > 1e-8) & small_den, 1.0, diff_right)
        diff_left = np.where(~small_num & (diff_left < -1e-8) & small_den, -1.0, diff_left)
        diff_right = np.where(~small_num & (diff_left < -1e-8) & small_den, 1.0, diff_right)

        slope_right[:, 1:-2] = self._slope_limiter(diff_left / diff_right)

        # Vectorized Left and Right extrapolated q-values at the boundary j+1/2
        # For nx=401, we want j from 1 to nx-3 = 397, so q_left shape is (3, 399)
        # q[:, 1:-2] has shape (3, 398), q[:, 2:-1] has shape (3, 399) - mismatch!
        # Correct indexing:
        indices = slice(1, self.nx - 2)  # j = 1 to nx-3
        q_left[:, indices] = self.q[:, indices] + 0.5 * slope_left[:, indices] * (
            self.q[:, indices.start + 1 : indices.stop + 1] - self.q[:, indices]
        )
        q_right[:, indices] = self.q[:, indices.start + 1 : indices.stop + 1] - 0.5 * slope_right[:, indices] * (
            self.q[:, indices.start + 2 : indices.stop + 2] - self.q[:, indices.start + 1 : indices.stop + 1]
        )

        # Boundary conditions
        q_right[:, 0] = self.q[:, 1] - 0.5 * slope_right[:, 1] * (self.q[:, 3] - self.q[:, 2])
        q_left[:, 0] = q_right[:, 0]
        q_left[:, self.nx - 2] = self.q[:, self.nx - 2] + 0.5 * slope_left[:, self.nx - 2] * (
            self.q[:, self.nx - 1] - self.q[:, self.nx - 2]
        )
        q_right[:, self.nx - 2] = q_left[:, self.nx - 2]

        # Vectorized Roe flux computation for all interfaces
        flux = self._flux_roe(q_left, q_right)

        dF = flux[:, 1:-1] - flux[:, 0:-2]
        return dF

    def cal_dt(self):
        """Calculate timestep using CFL condition"""
        # Compute primitive variables
        rho, u, p = self._cons2prim(self.q)
        a = np.sqrt(self.gamma * p / rho)  # Speed of sound

        # Maximum eigenvalue
        max_speed = np.max(np.abs(u) + a)
        dt = self.cfl * self.dx / max_speed
        return dt

    def step(self, dt):
        """Perform a single time step"""
        q0 = self.q.copy()
        dF = self._flux_muscl()

        # Update interior cells
        self.q[:, 1:-2] = q0[:, 1:-2] - dt / self.dx * dF

        # Boundary conditions (keep fixed)
        self.q[:, 0] = q0[:, 0]
        self.q[:, -1] = q0[:, -1]

        # Check for negative pressure
        _, _, p = self._cons2prim(self.q)

        if np.min(p) < 0:
            if self.verbose:
                print(f"Warning: Negative pressure detected at time {self.current_time}")

    def dump(self):
        """Save current state including data file and visualization"""
        # Create filename base
        file_base = os.path.join(self.dump_dir, f"res_{self.record_frame}")

        # Compute primitive variables
        rho, u, p = self._cons2prim(self.q)
        E = self.q[2] / rho

        # Save HDF5 data file
        with h5py.File(f"{file_base}.h5", "w") as f:
            f.create_dataset("x", data=self.x)
            f.create_dataset("rho", data=rho)
            f.create_dataset("u", data=u)
            f.create_dataset("p", data=p)
            f.create_dataset("E", data=E)
            f.create_dataset("time", data=self.current_time)

        # Create and save plot
        plt.figure(figsize=(10, 12))

        plt.subplot(4, 1, 1)
        plt.plot(self.x, rho, "k-", linewidth=2)
        plt.ylabel("Density (rho)")
        plt.tick_params(axis="x", bottom=False, labelbottom=False)
        plt.grid(True)
        plt.title(f"1D Euler Equations - Time = {self.current_time:.3f}")

        plt.subplot(4, 1, 2)
        plt.plot(self.x, u, "r-", linewidth=2)
        plt.ylabel("Velocity (u)")
        plt.tick_params(axis="x", bottom=False, labelbottom=False)
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(self.x, p, "b-", linewidth=2)
        plt.ylabel("Pressure (p)")
        plt.tick_params(axis="x", bottom=False, labelbottom=False)
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(self.x, E, "g-", linewidth=2)
        plt.ylabel("Energy (E)")
        plt.xlabel("Position (x)")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{file_base}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def post_process(self):
        """Post-processing: save metadata"""
        cost = self.num_steps * self.n_space
        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            meta = {
                "cost": cost,
                "cfl": float(self.cfl),
                "beta": float(self.beta),
                "k": float(self.k),
                "total_steps": int(self.num_steps),
                "gamma": float(self.gamma),
            }
            json.dump(meta, f, indent=4)
        if self.verbose:
            print(f"Run cost: {cost}")
