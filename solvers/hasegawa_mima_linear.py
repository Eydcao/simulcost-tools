import numpy as np
import h5py
import os
import json
import matplotlib.pyplot as plt
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
        self.case = cfg.case  # Initial condition case
        self.L = cfg.L  # Domain size
        self.v_star = cfg.v_star  # Diamagnetic drift velocity
        self.Dx = cfg.Dx  # Initial condition spatial scale

        # Tunable parameters
        self.N = cfg.N  # Grid resolution
        self.dt = cfg.dt  # Time step
        self.cg_atol = cfg.cg_atol  # CG solver absolute tolerance
        self.cg_maxiter = cfg.cg_maxiter  # CG solver max iterations

        # Method selection
        self.analytical = cfg.analytical  # Use analytical solution if True

        # Grid setup
        self.dx = self.dy = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)
        self.y = np.linspace(0, self.L, self.N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Performance tracking (initialize early)
        self.cg_iterations_total = 0
        self.cg_calls = 0
        self.matvec_operations = 0  # Track matrix-vector multiplications separately

        # Residual tracking for CG convergence analysis (limit to first 10 time steps)
        self.cg_residual_trajectories = []  # List of residual trajectories for each CG call
        self.max_tracked_steps = 10  # Only track first 10 simulation steps
        self.current_step = 0

        # Output directory
        if self.analytical:
            method_suffix = "_analytical"
            self.dump_dir = cfg.dump_dir + f"_N_{self.N}_dt_{self.dt:.2e}" + method_suffix
        else:
            method_suffix = "_numerical"
            self.dump_dir = cfg.dump_dir + f"_N_{self.N}_dt_{self.dt:.2e}_cg_{self.cg_atol:.2e}" + method_suffix
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Create analytical reference directory for numerical runs
        if not self.analytical:
            self.analytical_dir = cfg.dump_dir + f"_N_{self.N}_dt_{self.dt:.2e}_analytical"
            if not os.path.exists(self.analytical_dir):
                os.makedirs(self.analytical_dir)

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
        # Support multiple initial condition "cases" read from config as self.case.
        # If self.case is not provided, default to 'monopole'.
        case = self.case

        # Common aliases
        X = self.X
        Y = self.Y
        L = self.L
        Dx = self.Dx

        # Different predefined initial conditions
        if case == "monopole":
            # Gaussian monopole centered in the domain
            phi0 = 1e-1 * np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (2 * Dx**2))

        elif case == "dipole":
            # Gaussian dipole (odd in x)
            phi0 = 1e-1 * np.exp(-((X - L / 2) ** 2 + (Y - L / 2) ** 2) / (2 * Dx**2)) * ((X - L / 2) / Dx)

        elif case == "sinusoidal":
            # Pure sinusoidal in both directions
            phi0 = 1e-1 * np.sin(0.2 * X) * np.sin(0.3 * Y)

        elif case == "sin_x_gauss_y":
            # Sinusoidal in x, Gaussian in y
            phi0 = 1e-1 * np.sin(0.2 * X) * np.exp(-((Y - L / 2) ** 2) / (2 * Dx**2))

        elif case == "gauss_x_sin_y":
            # Gaussian in x, sinusoidal in y
            phi0 = 1e-1 * np.exp(-((X - L / 2) ** 2) / (2 * Dx**2)) * np.sin(0.2 * Y)

        else:
            # raise error for unknown case
            raise ValueError(f"Unknown initial condition case: {case}")

        return phi0

    def setup_numerical_method(self):
        """Setup sparse matrix operators for numerical solution"""
        # Construct 1D Laplacian with periodic BC
        e = np.ones(self.N)
        L1D = diags([e, -2 * e, e], offsets=[-1, 0, 1], shape=(self.N, self.N)).tolil()
        L1D[0, -1] = 1  # Periodic boundary
        L1D[-1, 0] = 1  # Periodic boundary
        L1D /= self.dx**2

        # 2D Laplacian operator
        L2D = kron(eye(self.N), L1D) + kron(L1D, eye(self.N))

        # Helmholtz operator: ∇² - I
        self.A_sparse = L2D - eye(self.N * self.N)

        # Derivative operator for ∂φ/∂y
        Dy = diags([1, -1], offsets=[1, -1], shape=(self.N, self.N)).tolil()
        Dy[0, -1] = -1  # Periodic boundary
        Dy[-1, 0] = 1  # Periodic boundary
        Dy /= 2 * self.dy

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
        # Use callback to count iterations precisely and track residuals
        current_iterations = 0
        residual_trajectory = []

        # Track residuals only for first few simulation steps
        track_residuals = self.current_step < self.max_tracked_steps

        def iteration_callback(x):
            nonlocal current_iterations, residual_trajectory
            current_iterations += 1

            if track_residuals:
                # Compute residual: ||A*x - b||
                residual_vec = self.A_sparse @ x - q_vec
                residual_norm = np.linalg.norm(residual_vec)
                residual_trajectory.append(residual_norm)

        phi_vec, info = cg(
            self.A_sparse, q_vec, atol=self.cg_atol, maxiter=self.cg_maxiter, callback=iteration_callback
        )

        # Only print CG info if verbose mode is enabled
        if self.verbose:
            print(f"CG solver info: {info}, iterations: {current_iterations}")

        # Store residual trajectory if tracking
        if track_residuals and residual_trajectory:
            self.cg_residual_trajectories.append(
                {
                    "step": self.current_step,
                    "cg_call": self.cg_calls,
                    "iterations": current_iterations,
                    "final_info": info,
                    "cg_atol": self.cg_atol,
                    "residual_trajectory": residual_trajectory.copy(),
                    "converged": (info == 0),
                }
            )

        # Track CG performance precisely
        self.cg_calls += 1
        self.cg_iterations_total += current_iterations
        if info == 0:
            # Successful convergence - use actual iteration count
            # self.cg_iterations_total += current_iterations
            pass
        else:
            # Failed convergence - info contains the number of iterations performed
            # self.cg_iterations_total += info
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

        # RHS: v_star * ∂φ/∂y (match reference implementation)
        return self.v_star * dphidy_vec

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

        # Increment step counter for residual tracking
        self.current_step += 1

    # def calculate_error_vs_analytical(self):
    #     """Calculate error compared to analytical solution (for numerical method)"""
    #     if self.analytical:
    #         return 0.0  # No error for analytical solution

    #     # Setup analytical method components if not already done
    #     if not hasattr(self, "KY"):
    #         self.setup_analytical_method()

    #     l2_norms = []
    #     for i, t in enumerate(self.actual_snapshot_times):
    #         phi_analytical = self.solve_analytical(t)
    #         phi_numerical = self.snapshots[i]
    #         diff = phi_numerical - phi_analytical
    #         l2 = np.sqrt(np.mean(diff**2))
    #         l2_norms.append(l2)

    # return np.mean(l2_norms)

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

        # Store snapshot for later processing
        self.snapshots.append(phi.copy())
        self.actual_snapshot_times.append(self.current_time)

        # Save to HDF5
        output_file = os.path.join(self.dump_dir, f"frame_{self.record_frame:04d}.h5")
        with h5py.File(output_file, "w") as f:
            f.create_dataset("phi", data=phi)
            f.create_dataset("coordinates_x", data=self.x)
            f.create_dataset("coordinates_y", data=self.y)
            f.attrs["time"] = self.current_time
            f.attrs["N"] = self.N
            f.attrs["dt"] = self.dt

        # For numerical runs, also save analytical solution for comparison
        if not self.analytical:
            analytical_output_file = os.path.join(self.analytical_dir, f"frame_{self.record_frame:04d}.h5")

            # Check if analytical solution already exists
            if os.path.exists(analytical_output_file):
                # Load existing analytical solution
                with h5py.File(analytical_output_file, "r") as f:
                    phi_analytical = f["phi"][:]
            else:
                # Compute and save analytical solution
                # Setup analytical method components if not already done
                if not hasattr(self, "KY"):
                    self.setup_analytical_method()
                phi_analytical = self.solve_analytical(self.current_time)

                # Save analytical solution to separate directory
                with h5py.File(analytical_output_file, "w") as f:
                    f.create_dataset("phi", data=phi_analytical)
                    f.create_dataset("coordinates_x", data=self.x)
                    f.create_dataset("coordinates_y", data=self.y)
                    f.attrs["time"] = self.current_time
                    f.attrs["N"] = self.N
                    f.attrs["dt"] = self.dt

            # Calculate error for inline display
            diff = phi - phi_analytical
            error = np.sqrt(np.mean(diff**2))
        else:
            # For analytical runs, set up plotting variables
            phi_analytical = phi
            diff = np.zeros_like(phi)
            error = 0.0

        # Optional plotting (field + analytical + difference, and power spectrum)
        # Prepare grids for plotting
        X, Y = np.meshgrid(self.x, self.y)

        # Field plot: phi, analytical, difference
        try:
            vlim = max(np.abs(phi).max(), np.abs(phi_analytical).max())
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

            im0 = axes[0].pcolormesh(X, Y, phi, cmap="RdBu", shading="auto", vmin=-vlim, vmax=vlim)
            axes[0].set_title(f"phi (t={self.current_time:.3f})")
            fig.colorbar(im0, ax=axes[0], label="phi")

            im1 = axes[1].pcolormesh(X, Y, phi_analytical, cmap="RdBu", shading="auto", vmin=-vlim, vmax=vlim)
            axes[1].set_title("analytical")
            fig.colorbar(im1, ax=axes[1], label="phi")

            diff_vlim = np.max(np.abs(diff))
            im2 = axes[2].pcolormesh(X, Y, diff, cmap="bwr", shading="auto", vmin=-diff_vlim, vmax=diff_vlim)
            axes[2].set_title(f"difference (L2={error:.2e})")
            fig.colorbar(im2, ax=axes[2], label="Δphi")

            plot_file = os.path.join(self.dump_dir, f"frame_{self.record_frame:04d}.png")
            fig.suptitle(f"Frame {self.record_frame}: t={self.current_time:.3f}")
            fig.savefig(plot_file, dpi=200)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: plotting failed: {e}")

        # Power spectrum plot (log power)
        try:
            phi_hat = np.fft.fft2(phi)
            phi_hat_anal = np.fft.fft2(phi_analytical)
            power = np.log10(np.abs(np.fft.fftshift(phi_hat)) ** 2 + 1e-15)
            power_anal = np.log10(np.abs(np.fft.fftshift(phi_hat_anal)) ** 2 + 1e-15)
            power_diff = power - power_anal

            kx = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi)
            ky = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.dy) * 2 * np.pi)
            KX, KY = np.meshgrid(kx, ky)

            fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            vmin = min(power.min(), power_anal.min())
            vmax = max(power.max(), power_anal.max())

            im0 = axes2[0].pcolormesh(KX, KY, power, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
            axes2[0].set_title("log10 Power (phi)")
            fig2.colorbar(im0, ax=axes2[0])

            im1 = axes2[1].pcolormesh(KX, KY, power_anal, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
            axes2[1].set_title("log10 Power (analytical)")
            fig2.colorbar(im1, ax=axes2[1])

            im2 = axes2[2].pcolormesh(KX, KY, power_diff, cmap="RdBu", shading="auto")
            axes2[2].set_title("Power difference")
            fig2.colorbar(im2, ax=axes2[2])

            ps_file = os.path.join(self.dump_dir, f"frame_{self.record_frame:04d}_spectrum.png")
            fig2.suptitle(f"Spectrum Frame {self.record_frame}: t={self.current_time:.3f}")
            fig2.savefig(ps_file, dpi=200)
            plt.close(fig2)
        except Exception as e:
            print(f"Warning: spectrum plotting failed: {e}")

            if self.verbose:
                # method_name = "Analytical" if self.analytical else "Numerical"
                print(f"Frame {self.record_frame}: t={self.current_time:.3f}, error={error:.6e}")

    def post_process(self):
        """Post-processing: save metadata"""
        cost = self.estimate_cost()

        # Main metadata (compact, for production use)
        meta = {
            "cost": cost,
            "N": int(self.N),
            "dt": float(self.dt),
            "cg_atol": float(self.cg_atol),
            "cg_maxiter": int(self.cg_maxiter),
            "analytical": self.analytical,
            "n_steps": self.num_steps,
            "cg_iterations_total": self.cg_iterations_total,
            "cg_calls": self.cg_calls,
            "matvec_operations": self.matvec_operations,
            "wall_time_total": float(self.wall_time_total),
            "wall_time_exceeded": bool(self.wall_time_exceeded),
        }

        # For numerical runs, save analytical directory path in metadata
        if not self.analytical:
            meta["analytical_reference_dir"] = self.analytical_dir

        # Save main metadata
        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        # Save verbose metadata with CG trajectories if verbose mode is on
        if self.verbose and self.cg_residual_trajectories:
            verbose_meta = meta.copy()
            verbose_meta["cg_residual_trajectories"] = self.cg_residual_trajectories
            with open(os.path.join(self.dump_dir, "verbose_meta.json"), "w") as f:
                json.dump(verbose_meta, f, indent=4)
            print(f"Saved verbose metadata with {len(self.cg_residual_trajectories)} CG residual trajectories")

        # Save detailed residual analysis if we have trajectories
        if self.cg_residual_trajectories:
            self.save_residual_analysis()

        if self.verbose:
            print(f"Run cost: {cost}")

    def save_residual_analysis(self):
        """Save detailed CG residual trajectory analysis and plots"""
        if not self.cg_residual_trajectories:
            return

        # Create residual analysis plot
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                f"CG Residual Analysis (N={self.N}, dt={self.dt:.1e}, cg_atol={self.cg_atol:.1e})", fontsize=14
            )

            # Plot 1: Residual trajectories vs iterations
            ax1 = axes[0, 0]
            for i, traj_data in enumerate(self.cg_residual_trajectories):
                if i < 5:  # Only show first 5 trajectories to avoid clutter
                    residuals = traj_data["residual_trajectory"]
                    ax1.semilogy(residuals, label=f'Step {traj_data["step"]}, Call {traj_data["cg_call"]}')

            ax1.axhline(y=self.cg_atol, color="red", linestyle="--", label=f"atol={self.cg_atol:.1e}")
            ax1.set_xlabel("CG Iteration")
            ax1.set_ylabel("Residual Norm")
            ax1.set_title("Residual Trajectories (First 5 Calls)")
            ax1.legend()
            ax1.grid(True)

            # Plot 2: Convergence rate analysis
            ax2 = axes[0, 1]
            convergence_rates = []
            step_numbers = []
            for traj_data in self.cg_residual_trajectories:
                residuals = traj_data["residual_trajectory"]
                if len(residuals) > 1:
                    # Estimate convergence rate from residual drop
                    log_residuals = np.log10(np.array(residuals))
                    if len(log_residuals) > 3:
                        rate = -(log_residuals[-1] - log_residuals[1]) / (len(log_residuals) - 2)
                        convergence_rates.append(rate)
                        step_numbers.append(traj_data["step"])

            if convergence_rates:
                ax2.plot(step_numbers, convergence_rates, "bo-")
                ax2.set_xlabel("Simulation Step")
                ax2.set_ylabel("Log10 Convergence Rate")
                ax2.set_title("CG Convergence Rate vs Time")
                ax2.grid(True)

            # Plot 3: Iteration count vs step
            ax3 = axes[1, 0]
            iterations_per_call = [traj["iterations"] for traj in self.cg_residual_trajectories]
            steps_per_call = [traj["step"] for traj in self.cg_residual_trajectories]
            ax3.plot(steps_per_call, iterations_per_call, "go-")
            ax3.set_xlabel("Simulation Step")
            ax3.set_ylabel("CG Iterations")
            ax3.set_title("CG Iterations Required vs Time")
            ax3.grid(True)

            # Plot 4: Final residual vs atol
            ax4 = axes[1, 1]
            final_residuals = []
            atol_values = []
            for traj_data in self.cg_residual_trajectories:
                if traj_data["residual_trajectory"]:
                    final_residuals.append(traj_data["residual_trajectory"][-1])
                    atol_values.append(traj_data["cg_atol"])

            if final_residuals:
                ax4.loglog(atol_values, final_residuals, "ro", label="Final Residual")
                ax4.plot(
                    [min(atol_values), max(atol_values)],
                    [min(atol_values), max(atol_values)],
                    "b--",
                    label="y=x (ideal)",
                )
                ax4.set_xlabel("CG atol")
                ax4.set_ylabel("Final Residual")
                ax4.set_title("Achieved vs Requested Tolerance")
                ax4.legend()
                ax4.grid(True)

            plt.tight_layout()
            residual_plot_file = os.path.join(self.dump_dir, "cg_residual_analysis.png")
            plt.savefig(residual_plot_file, dpi=200, bbox_inches="tight")
            plt.close()

            # Save summary statistics
            summary_file = os.path.join(self.dump_dir, "cg_residual_summary.txt")
            with open(summary_file, "w") as f:
                f.write(f"CG Residual Analysis Summary\n")
                f.write(f"============================\n\n")
                f.write(f"Simulation Parameters:\n")
                f.write(f"  N = {self.N}\n")
                f.write(f"  dt = {self.dt:.2e}\n")
                f.write(f"  cg_atol = {self.cg_atol:.2e}\n")
                f.write(f"  cg_maxiter = {self.cg_maxiter}\n\n")

                f.write(f"CG Performance Summary:\n")
                f.write(f"  Total CG calls tracked: {len(self.cg_residual_trajectories)}\n")

                if self.cg_residual_trajectories:
                    iterations_list = [traj["iterations"] for traj in self.cg_residual_trajectories]
                    f.write(f"  Average iterations per call: {np.mean(iterations_list):.1f}\n")
                    f.write(f"  Min/Max iterations: {np.min(iterations_list)}/{np.max(iterations_list)}\n")

                    final_residuals_list = [
                        traj["residual_trajectory"][-1]
                        for traj in self.cg_residual_trajectories
                        if traj["residual_trajectory"]
                    ]
                    if final_residuals_list:
                        f.write(f"  Average final residual: {np.mean(final_residuals_list):.2e}\n")
                        f.write(
                            f"  Min/Max final residual: {np.min(final_residuals_list):.2e}/{np.max(final_residuals_list):.2e}\n"
                        )

                f.write(f"\nDetailed Call Information:\n")
                for i, traj_data in enumerate(self.cg_residual_trajectories):
                    f.write(f"  Call {i+1} (Step {traj_data['step']}):\n")
                    f.write(f"    Iterations: {traj_data['iterations']}\n")
                    f.write(f"    Converged: {traj_data['converged']}\n")
                    if traj_data["residual_trajectory"]:
                        f.write(f"    Initial residual: {traj_data['residual_trajectory'][0]:.2e}\n")
                        f.write(f"    Final residual: {traj_data['residual_trajectory'][-1]:.2e}\n")
                        if len(traj_data["residual_trajectory"]) > 1:
                            reduction = traj_data["residual_trajectory"][0] / traj_data["residual_trajectory"][-1]
                            f.write(f"    Residual reduction: {reduction:.1e}\n")
                    f.write(f"\n")

        except Exception as e:
            print(f"Warning: CG residual analysis failed: {e}")
