import numpy as np
import h5py
import os
import json
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
from .base_solver import SIMULATOR


class HasegawaMimaNonlinear(SIMULATOR):
    """
    Nonlinear Hasegawa-Mima equation solver using pseudo-spectral method.

    Solves: ∂q/∂t + {φ, q} + v_star * ∂φ/∂y = 0
    where q = ∇²φ - φ (generalized vorticity)
    and {φ, q} is the Poisson bracket

    Uses pseudo-spectral method with:
    - 2D FFT for spatial derivatives
    - 2/3 rule dealiasing for nonlinear terms
    - RK4 time integration
    """

    def __init__(self, verbose, cfg):
        # Environmental/Physics parameters (fixed)
        self.case = cfg.case  # Initial condition case
        self.L = cfg.L  # Domain size
        self.v_star = cfg.v_star  # Diamagnetic drift velocity
        self.Dx = cfg.Dx  # Initial condition spatial scale

        # Tunable parameters
        self.N = cfg.N  # Grid resolution
        self.dt = cfg.dt  # Time step

        # Fixed dealiasing parameter
        self.dealias_ratio = 2/3  # Stronger dealiasing with smaller ratio

        # Grid setup
        self.dx = self.dy = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)
        self.y = np.linspace(0, self.L, self.N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Spectral grid setup
        self.kx = fftfreq(self.N, d=self.dx) * 2 * np.pi
        self.ky = fftfreq(self.N, d=self.dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.k2 = self.KX**2 + self.KY**2
        self.A_fft = -(1 + self.k2)  # Helmholtz operator in Fourier space

        # Dealiasing mask
        self.dealias = self.create_dealias_mask()

        # Performance tracking
        self.fft_operations = 0  # Track FFT operations
        self.poisson_bracket_calls = 0  # Track nonlinear term evaluations

        # Output directory
        self.dump_dir = cfg.dump_dir + f"_N_{self.N}_dt_{self.dt:.2e}_nonlinear"
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Initialize solution
        self.phi0 = self.initialize_condition()
        self.phi0_hat = fft2(self.phi0)
        self.q_hat = self.A_fft * self.phi0_hat  # Initial vorticity in spectral space
        self.fft_operations += 2  # Count initial FFTs

        # Storage for snapshots
        self.snapshots = []
        self.actual_snapshot_times = []

        # Base initialization (handles record_dt, end_frame -> end_time)
        super().__init__(verbose, cfg)

    def create_dealias_mask(self):
        """Create 2/3 rule dealiasing mask"""
        def dealias_mask_1d(N):
            cutoff = int(N * self.dealias_ratio // 2)
            mask = np.zeros(N)
            mask[:cutoff+1] = 1
            mask[-cutoff:] = 1
            return mask

        mask_x = dealias_mask_1d(self.N)
        mask_y = dealias_mask_1d(self.N)
        return np.outer(mask_y, mask_x)

    def initialize_condition(self):
        """Initialize with various initial conditions"""
        case = self.case

        # Common aliases
        X = self.X
        Y = self.Y
        L = self.L
        Dx = self.Dx

        # Different predefined initial conditions (reduced amplitude for stability)
        if case == "monopole":
            # Gaussian monopole centered in the domain
            phi0 = 0.1 * np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (2 * Dx**2))

        elif case == "dipole":
            # Gaussian dipole (odd in x)
            phi0 = 0.1 * np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (2 * Dx**2)) * ((X - L / 2) / Dx)

        elif case == "sinusoidal":
            # Pure sinusoidal in both directions
            phi0 = 0.1 * np.sin(0.2 * X) * np.sin(0.3 * Y)

        elif case == "sin_x_gauss_y":
            # Sinusoidal in x, Gaussian in y
            phi0 = 0.1 * np.sin(0.2 * X) * np.exp(-((Y - L / 2) ** 2) / (2 * Dx**2))

        elif case == "gauss_x_sin_y":
            # Gaussian in x, sinusoidal in y
            phi0 = 0.1 * np.exp(-((X - L / 2) ** 2) / (2 * Dx**2)) * np.sin(0.2 * Y)

        else:
            raise ValueError(f"Unknown case: {case}")

        return phi0

    def poisson_bracket_dealiased(self, phi_hat, q_hat):
        """
        Compute {phi, q} using spectral method with proper dealiasing
        Applies the 2/3 rule dealiasing only to the nonlinear Poisson bracket
        """
        # Compute derivatives in spectral space
        dphidx_hat = 1j * self.KX * phi_hat
        dphidy_hat = 1j * self.KY * phi_hat
        dqdx_hat = 1j * self.KX * q_hat
        dqdy_hat = 1j * self.KY * q_hat

        # Transform to physical space
        dphidx = ifft2(dphidx_hat).real
        dphidy = ifft2(dphidy_hat).real
        dqdx = ifft2(dqdx_hat).real
        dqdy = ifft2(dqdy_hat).real

        # Track FFT operations
        self.fft_operations += 8  # 4 forward + 4 inverse FFTs

        # Compute Jacobian in physical space
        jacobian = dphidx * dqdy - dphidy * dqdx

        # Transform back to spectral space with dealiasing
        jacobian_hat = fft2(jacobian) * self.dealias
        self.fft_operations += 1  # One more forward FFT

        self.poisson_bracket_calls += 1
        return jacobian_hat

    def rhs(self, q_hat):
        """Compute RHS of nonlinear Hasegawa-Mima equation in spectral space"""
        # Compute phi from q using Poisson equation in spectral space
        phi_hat = np.copy(q_hat)
        # Avoid division by zero at k=0
        nonzero_mask = np.abs(self.A_fft) > 1e-14
        phi_hat[nonzero_mask] = q_hat[nonzero_mask] / self.A_fft[nonzero_mask]
        phi_hat[~nonzero_mask] = 0.0

        # Compute dealiased Jacobian in spectral space
        jacobian_hat = self.poisson_bracket_dealiased(phi_hat, q_hat)

        # Compute y derivative (linear term, no dealiasing needed)
        dphidy_hat = 1j * self.KY * phi_hat

        # Compute RHS in spectral space
        rhs_hat = -jacobian_hat + self.v_star * dphidy_hat

        return rhs_hat

    def pre_process(self):
        """Initialize simulation before time stepping"""
        # Record initial condition
        self.snapshots.append(self.phi0.copy())
        self.actual_snapshot_times.append(0.0)

        if self.verbose:
            print(f"Nonlinear Hasegawa-Mima simulation initialized")
            print(f"Grid: {self.N}x{self.N}, Domain: {self.L:.2f}x{self.L:.2f}")
            print(f"Time step: {self.dt:.6f}, End time: {self.end_time:.2f}")
            print(f"Dealiasing ratio: {self.dealias_ratio}")

    def cal_dt(self):
        """Return the configured timestep"""
        return self.dt

    def step(self, dt):
        """Single RK4 step for the nonlinear Hasegawa-Mima equation"""
        # RK4 integration step
        r1 = self.rhs(self.q_hat)
        r2 = self.rhs(self.q_hat + 0.5 * dt * r1)
        r3 = self.rhs(self.q_hat + 0.5 * dt * r2)
        r4 = self.rhs(self.q_hat + dt * r3)

        self.q_hat += dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4)

    def dump(self):
        """Save simulation state at current time"""
        # Compute current phi from q_hat
        phi_hat = self.q_hat / self.A_fft
        phi = ifft2(phi_hat).real
        self.fft_operations += 1  # Count inverse FFT

        # Store snapshot
        self.snapshots.append(phi.copy())
        self.actual_snapshot_times.append(self.current_time)

        # Save to HDF5 file
        filename = f"frame_{self.record_frame:04d}.h5"
        filepath = os.path.join(self.dump_dir, filename)

        with h5py.File(filepath, 'w') as f:
            f.create_dataset('phi', data=phi)
            f.create_dataset('coordinates_x', data=self.x)
            f.create_dataset('coordinates_y', data=self.y)

            # Store metadata as attributes
            f.attrs['time'] = self.current_time
            f.attrs['N'] = self.N
            f.attrs['dt'] = self.dt
            f.attrs['dealias_ratio'] = self.dealias_ratio
            f.attrs['fft_operations'] = self.fft_operations
            f.attrs['poisson_bracket_calls'] = self.poisson_bracket_calls

        # Create visualization similar to linear case
        try:
            X, Y = np.meshgrid(self.x, self.y)
            vlim = np.max(np.abs(phi))

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            im = ax.pcolormesh(X, Y, phi, cmap='RdBu', shading='auto', vmin=-vlim, vmax=vlim)
            ax.set_title(f'Nonlinear HM φ (t={self.current_time:.3f})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax, label='φ')

            plot_file = os.path.join(self.dump_dir, f"frame_{self.record_frame:04d}.png")
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Important: close figure to prevent display
        except Exception as e:
            if self.verbose:
                print(f"Warning: plotting failed: {e}")

        if self.verbose:
            print(f"Frame {self.record_frame} saved at t={self.current_time:.4f}")

    def post_process(self):
        """Finalize simulation and save metadata"""
        # Calculate computational cost estimate
        # Cost is dominated by FFT operations
        cost = self.fft_operations * self.N**2 * np.log2(self.N**2)

        # Save metadata
        meta = {
            'cost': float(cost),
            'n_steps': self.num_steps,
            'fft_operations': self.fft_operations,
            'poisson_bracket_calls': self.poisson_bracket_calls,
            'end_time': self.current_time,
            'N': self.N,
            'dt': self.dt,
            'dealias_ratio': self.dealias_ratio,
            'case': self.case,
            'L': self.L,
            'v_star': self.v_star,
            'Dx': self.Dx
        }

        meta_path = os.path.join(self.dump_dir, 'meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        if self.verbose:
            print(f"Simulation completed: {self.num_steps} steps, cost={cost:.2e}")
            print(f"FFT operations: {self.fft_operations}")
            print(f"Poisson bracket calls: {self.poisson_bracket_calls}")