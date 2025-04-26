import numpy as np
import h5py
import os
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from base_solver import SIMULATOR


class Heat1D(SIMULATOR):
    """Heat transfer in 1D with Crank-Nicolson scheme, left boundary with convection to T_inf, right boundary dirichlet."""
    def __init__(self, verbose, cfg):
        # Physical parameters from cfg
        self.L = cfg.L  # Wall thickness
        self.k = cfg.k  # Thermal conductivity
        self.h = cfg.h  # Convection coefficient
        self.rho = cfg.rho  # Density
        self.cp = cfg.cp  # Specific heat
        self.T_inf = cfg.T_inf  # Ambient temperature
        self.T_init = cfg.T_init  # Initial temperature

        # Numerical parameters
        self.n_space = cfg.n_space  # Number of spatial points
        self.alpha = self.k / (self.rho * self.cp)  # Thermal diffusivity
        self.cfl = cfg.cfl  # CFL number

        # dir of dump
        self.dump_dir = cfg.dump_dir
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Initialize spatial grid
        self.dx = self.L / self.n_space
        self.nx = self.n_space + 1
        self.x = np.linspace(0, self.L, self.nx)

        # Base initialization
        super().__init__(verbose, cfg)

    def init_fields(self):
        """Initialize any additional fields needed"""
        # Initialize temperature field
        self.T = self.T_init * np.ones(self.nx)

        # LU cache, using dt as key
        self.LU_pivots_cache = {}

    def _calculate_and_factorize_crank_nicolson_matrix(self, dt):
        """Calculate and factorize the Crank-Nicolson matrix."""
        r = self.alpha * dt / (2 * self.dx**2)

        # I - 1/2 dt / dx^2 laplacian
        main_diag = np.ones(self.nx) * (1 + 2 * r)
        off_diag_right = np.ones(self.nx - 1) * -r
        off_diag_left = np.ones(self.nx - 1) * -r

        # Boundary conditions
        # Left boundary: convection to T_inf
        main_diag[0] = 1 + self.h * self.dx / self.k
        off_diag_right[0] = -1
        # Right boundary: adiabatic
        main_diag[-1] = 1
        off_diag_left[-1] = -1

        # Construct the matrix
        A = diags([main_diag, off_diag_right, off_diag_left], [0, 1, -1], format="csr")

        # Perform LU factorization
        return splu(A.tocsc())

    def cal_dt(self):
        """Calculate base timestep using CFL condition"""
        # Conservative CFL condition for diffusion
        max_dt = self.cfl * self.dx**2 / (2 * self.alpha)
        return max_dt

    def _calculate_right_hand_side(self, T_old, dt):
        """Calculate the right hand side of the Crank-Nicolson scheme."""
        r = self.alpha * dt / (2 * self.dx**2)

        # Calculate the central [1:-1] part
        b = T_old.copy()
        b[1:-1] += r * (T_old[:-2] - 2 * T_old[1:-1] + T_old[2:])

        # Boundary conditions
        # Left boundary: convection to T_inf
        b[0] = self.h * self.dx * self.T_inf / self.k
        # Right boundary: adiabatic
        b[-1] = 0

        return b

    def step(self, dt):
        """Perform a single time step"""
        # Calculate system and right hand side
        if dt not in self.LU_pivots_cache:
            LU_pivots = self._calculate_and_factorize_crank_nicolson_matrix(dt)
            self.LU_pivots_cache[dt] = LU_pivots
        else:
            LU_pivots = self.LU_pivots_cache[dt]
        b = self._calculate_right_hand_side(self.T, dt)

        # Solve the linear system
        self.T = LU_pivots.solve(b)

    def dump(self):
        """Save current state"""
        with h5py.File(os.path.join(self.dump_dir, f"res_{self.record_frame}.h5"), "w") as f:
            f.create_dataset("x", data=self.x)
            f.create_dataset("T", data=self.T)
            f.create_dataset("time", data=self.current_time)
