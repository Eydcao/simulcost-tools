import torch
import h5py
import os
from .base_solver import SIMULATOR
import numpy as np


class Heat1D(SIMULATOR):
    """Heat transfer in 1D with Crank-Nicolson scheme, left boundary with convection to T_inf, right boundary adiabatic."""

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
        # NOTE append dump dir with the tunnable parameter cff, and nx
        self.dump_dir = cfg.dump_dir + f"_cfl_{self.cfl}_nx_{self.n_space}/"
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Initialize spatial grid
        self.dx = self.L / self.n_space
        self.nx = self.n_space + 1
        self.x = torch.linspace(0, self.L, self.nx)

        # Base initialization
        super().__init__(verbose, cfg)

    def init_fields(self):
        """Initialize any additional fields needed"""
        # Initialize temperature field
        self.T = self.T_init * torch.ones(self.nx)

    def cal_dt(self):
        """Calculate base timestep using CFL condition"""
        # Conservative CFL condition for diffusion
        max_dt = self.cfl * self.dx**2 / (2 * self.alpha)
        return max_dt

    def step(self, dt):
        """Perform a single time step"""
        # explicit form
        T_new = self.T.clone()
        # inner node
        T_new[1:-1] += self.alpha * dt / (self.dx**2) * (T_new[:-2] - 2 * T_new[1:-1] + T_new[2:])
        # right node is adiabatic
        T_new[-1] = T_new[-2]
        # left node is convective (enforce gradient)
        T_new[0] = (self.dx / self.k * T_new[1] + self.h * self.T_inf) / (self.dx / self.k + self.h)
        self.T = T_new

    def dump(self):
        """Save current state"""
        with h5py.File(os.path.join(self.dump_dir, f"res_{self.record_frame}.h5"), "w") as f:
            f.create_dataset("x", data=self.x.numpy())
            f.create_dataset("T", data=self.T.numpy())
            f.create_dataset("time", data=self.current_time)
