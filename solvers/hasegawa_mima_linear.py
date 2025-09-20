import numpy as np
import h5py
import os
import json
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import cg
from .base_solver import SIMULATOR


class HasegawaMimaLinear(SIMULATOR):
    """
    Linear Hasegawa-Mima equation solver with both analytical and numerical methods.

    Solves: ∂q/∂t + v_star * ∂φ/∂y = 0
    where q = ∇²φ - φ (generalized vorticity)

    Supports both:
    1. Numerical solution via RK4 + sparse matrix inversion (CG solver)
    2. Analytical solution via spectral method (for comparison)
    """

    def __init__(self, verbose, cfg):
        # Environmental/Physics parameters (fixed)
        self.L = cfg.L  # Domain size
        self.v_star = cfg.v_star  # Diamagnetic drift velocity
        self.Dx = cfg.Dx  # Initial condition spatial scale

        # Tunable parameters
        self.N = cfg.N  # Grid resolution
        self.dt = cfg.dt  # Time step
        self.cg_atol = cfg.cg_atol  # CG solver absolute tolerance
        self.cg_maxiter = cfg.cg_maxiter  # CG solver max iterations

        # Method selection
        self.analytical = getattr(cfg, 'analytical', False)

        # Grid setup
        self.dx = self.dy = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)
        self.y = np.linspace(0, self.L, self.N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Performance tracking (initialize early)
        self.cg_iterations_total = 0
        self.cg_calls = 0
        self.matvec_operations = 0  # Track matrix-vector multiplications separately

        # Output directory
        method_suffix = "_analytical" if self.analytical else "_numerical"
        self.dump_dir = cfg.dump_dir + f"_N_{self.N}_dt_{self.dt:.2e}" + method_suffix
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Initialize solution
        self.phi0 = self.initialize_condition()

        # Setup method components
        if not self.analytical:
            self.setup_numerical_method()
            # For numerical method, initialize state
            self.q_vec = self.A_sparse @ self.phi0.ravel()
            self.matvec_operations += 1  # Count initial A*phi operation
        else:
            self.setup_analytical_method()

        # Storage for snapshots (needed for error calculation)
        self.snapshots = []
        self.actual_snapshot_times = []

        # Base initialization (handles record_dt, end_frame -> end_time)
        super().__init__(verbose, cfg)

    def initialize_condition(self):
        """Initialize with Gaussian blob (monopole)"""
        return 1e-1 * np.exp(-((self.X - self.L/2)**2 + (self.Y - self.L/2)**2) / (2*self.Dx**2))

    def setup_numerical_method(self):
        """Setup sparse matrix operators for numerical solution"""
        # Construct 1D Laplacian with periodic BC
        e = np.ones(self.N)
        L1D = diags([e, -2*e, e], offsets=[-1, 0, 1], shape=(self.N, self.N)).tolil()
        L1D[0, -1] = 1  # Periodic boundary
        L1D[-1, 0] = 1  # Periodic boundary
        L1D /= self.dx**2

        # 2D Laplacian operator
        L2D = kron(eye(self.N), L1D) + kron(L1D, eye(self.N))

        # Helmholtz operator: ∇² - I
        self.A_sparse = L2D - eye(self.N*self.N)

        # Derivative operator for ∂φ/∂y
        Dy = diags([1, -1], offsets=[1, -1], shape=(self.N, self.N)).tolil()
        Dy[0, -1] = -1  # Periodic boundary
        Dy[-1, 0] = 1   # Periodic boundary
        Dy /= (2 * self.dy)

        self.Dy_sparse = kron(Dy, eye(self.N))

    def setup_analytical_method(self):
        """Setup spectral method for analytical solution"""
        # Wavenumbers for FFT
        kx = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.N, d=self.dy) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.k2 = self.KX**2 + self.KY**2
        self.A_fft = -(1 + self.k2)

        # Initial condition in spectral space
        self.phi0_hat = np.fft.fft2(self.phi0)

    def solve_helmholtz_cg(self, q_vec):
        """Solve Helmholtz equation using conjugate gradient"""
        # Use callback to count iterations precisely
        current_iterations = 0
        def iteration_callback(x):
            nonlocal current_iterations
            current_iterations += 1

        phi_vec, info = cg(self.A_sparse, q_vec,
                          atol=self.cg_atol, maxiter=self.cg_maxiter,
                          callback=iteration_callback)

        # Track CG performance precisely
        self.cg_calls += 1
        if info == 0:
            # Successful convergence - use actual iteration count
            self.cg_iterations_total += current_iterations
        else:
            # Failed convergence - info contains the number of iterations performed
            self.cg_iterations_total += info
            if self.verbose:
                print(f"Warning: CG solver did not converge after {info} iterations")

        return phi_vec

    def rhs_numerical(self, q_vec):
        """Compute RHS for numerical method"""
        # Solve for phi from q
        phi_vec = self.solve_helmholtz_cg(q_vec)

        # Compute ∂φ/∂y (count this matvec operation)
        dphidy_vec = self.Dy_sparse @ phi_vec
        self.matvec_operations += 1

        # RHS: -v_star * ∂φ/∂y (negative sign from the equation)
        return -self.v_star * dphidy_vec

    def step_numerical(self, q_vec, dt):
        """RK4 step for numerical method"""
        r1 = self.rhs_numerical(q_vec)
        r2 = self.rhs_numerical(q_vec + 0.5 * dt * r1)
        r3 = self.rhs_numerical(q_vec + 0.5 * dt * r2)
        r4 = self.rhs_numerical(q_vec + dt * r3)

        return q_vec + dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4)

    def solve_analytical(self, t):
        """Analytical solution using spectral method"""
        # Phase evolution: exp(-i * v_star * ky * t / A_fft)
        # Note: A_fft = -(1 + k²), so we get exp(i * v_star * ky * t / (1 + k²))
        phase = np.exp(1j * self.v_star * self.KY * t / self.A_fft)
        phi_hat_t = self.phi0_hat * phase
        return np.real(np.fft.ifft2(phi_hat_t))

    def cal_dt(self):
        """Return the timestep for the simulation"""
        return self.dt

    def step(self, dt):
        """Advance simulation by one timestep"""
        if self.analytical:
            # For analytical method, we don't actually step
            # The solution is computed directly in dump()
            pass
        else:
            # Numerical method: RK4 step
            self.q_vec = self.step_numerical(self.q_vec, dt)

    def calculate_error_vs_analytical(self):
        """Calculate error compared to analytical solution (for numerical method)"""
        if self.analytical:
            return 0.0  # No error for analytical solution

        # Setup analytical method components if not already done
        if not hasattr(self, 'KY'):
            self.setup_analytical_method()

        l2_norms = []
        for i, t in enumerate(self.actual_snapshot_times):
            phi_analytical = self.solve_analytical(t)
            phi_numerical = self.snapshots[i]
            diff = phi_numerical - phi_analytical
            l2 = np.sqrt(np.mean(diff**2))
            l2_norms.append(l2)

        return np.mean(l2_norms)

    def estimate_cost(self):
        """Estimate computational cost"""
        if self.analytical:
            # Analytical: FFT operations at each output time
            n_outputs = int(self.end_time / self.record_dt) + 1  # Number of recordings
            fft_cost = n_outputs * self.N**2 * np.log2(self.N**2)  # 2D FFT
            return int(fft_cost)
        else:
            # Numerical: precise counting of operations
            # CG solver cost: each CG iteration costs roughly N^2 operations
            cg_cost = self.cg_iterations_total * self.N**2

            # Sparse matrix-vector multiply cost: each operation costs roughly N^2
            matvec_cost = self.matvec_operations * self.N**2

            total_cost = cg_cost + matvec_cost
            return int(total_cost)

    def pre_process(self):
        """Initialize simulation"""
        pass


    def call_back(self):
        """Called after recording"""
        pass

    def dump(self):
        """Save simulation state at current_time (called by base solver)"""
        # Get current solution
        if self.analytical:
            phi = self.solve_analytical(self.current_time)
        else:
            phi_vec = self.solve_helmholtz_cg(self.q_vec)
            phi = phi_vec.reshape((self.N, self.N))

        # Calculate error vs analytical (if numerical)
        if self.analytical:
            error = 0.0
        else:
            # Setup analytical method components if not already done
            if not hasattr(self, 'KY'):
                self.setup_analytical_method()
            phi_analytical = self.solve_analytical(self.current_time)
            diff = phi - phi_analytical
            error = np.sqrt(np.mean(diff**2))

        # Store snapshot for later processing
        self.snapshots.append(phi.copy())
        self.actual_snapshot_times.append(self.current_time)

        # Save to HDF5
        output_file = os.path.join(self.dump_dir, f"frame_{self.record_frame:04d}.h5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('phi', data=phi)
            f.create_dataset('coordinates_x', data=self.x)
            f.create_dataset('coordinates_y', data=self.y)
            f.attrs['time'] = self.current_time
            f.attrs['N'] = self.N
            f.attrs['dt'] = self.dt
            f.attrs['analytical'] = self.analytical
            f.attrs['error'] = error

        # Save to JSON for easy access
        results = {
            'phi': phi.tolist(),
            'coordinates_x': self.x.tolist(),
            'coordinates_y': self.y.tolist(),
            'time': self.current_time,
            'error': error,
            'parameters': {
                'N': self.N,
                'dt': self.dt,
                'L': self.L,
                'v_star': self.v_star,
                'Dx': self.Dx,
                'cg_atol': self.cg_atol,
                'cg_maxiter': self.cg_maxiter,
                'analytical': self.analytical
            }
        }

        json_file = os.path.join(self.dump_dir, f"frame_{self.record_frame:04d}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            method_name = "Analytical" if self.analytical else "Numerical"
            print(f"Frame {self.record_frame}: t={self.current_time:.3f}, error={error:.6e}")

    def post_process(self):
        """Post-processing: save metadata"""
        cost = self.estimate_cost()
        error = self.calculate_error_vs_analytical()

        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            meta = {
                "cost": cost,
                "error": error,
                "N": int(self.N),
                "dt": float(self.dt),
                "cg_atol": float(self.cg_atol),
                "cg_maxiter": int(self.cg_maxiter),
                "analytical": self.analytical,
                "solve_time": getattr(self, 'solve_time', 0.0),
                "n_steps": self.num_steps,
                "cg_iterations_total": self.cg_iterations_total,
                "cg_calls": self.cg_calls,
                "matvec_operations": self.matvec_operations
            }
            json.dump(meta, f, indent=4)

        if self.verbose:
            print(f"Run cost: {cost}")