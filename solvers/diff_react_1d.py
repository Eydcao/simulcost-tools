import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import json
from .base_solver import SIMULATOR
from .utils import format_param_for_path
from scipy import linalg
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve

class NewtonOptimizer:
    def __init__(
        self,
        tol,
        max_iter,
        min_step,
        verbose,
        initial_step_guess=1.0,
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.min_step = min_step
        self.verbose = verbose
        self.initial_step_guess = initial_step_guess

    def assemble_system(self, u, **kwargs):
        """
        Assemble Jacobian matrix for current solution u
        Args:
            u: Current solution vector
            **kwargs: Additional problem-specific parameters
        Returns:
            jacobian: 2D numpy array
        """
        raise NotImplementedError("Subclasses must implement assemble_system()")

    def calc_residual(self, u, **kwargs):
        """
        Calculate residual vector for current solution u
        Args:
            u: Current solution vector
            **kwargs: Additional problem-specific parameters
        Returns:
            residual: 1D numpy array
        """
        raise NotImplementedError("Subclasses must implement calc_residual()")

    def solve_direction(self, jacobian, residual): # N_res
        """
        Solve for optimization direction using factorization
        """
        try:
            if isinstance(jacobian, np.ndarray):
                direction = -linalg.solve(jacobian, residual)
            elif isinstance(jacobian, csr_matrix):
                direction = -spsolve(jacobian, residual) # N_res
            else:
                raise ValueError("Unsupported Jacobian matrix format")
        except (np.linalg.LinAlgError, linalg.ArpackNoConvergence):
            raise ValueError("Failed to solve system - Jacobian may be singular")
        return direction

    def residual_norm(self, residual):
        """
        Calculate L2 norm of residual
        """
        return np.max(residual**2)

    def line_search(self, u, direction, initial_residual_norm, **kwargs): # iter_line_search * N_res
        """
        Perform line search with greedy approach
        """
        alpha = self.initial_step_guess
        current_u = u
        min_alpha = alpha
        min_residual_norm = float("inf")
        line_search_iters = 0

        while alpha > self.min_step:
            # Try step
            trial_u = current_u + alpha * direction
            trial_residual = self.calc_residual(trial_u, **kwargs)
            trial_residual_norm = self.residual_norm(trial_residual)
            line_search_iters += 1

            # Track minimum residual norm and associated alpha
            if trial_residual_norm < min_residual_norm:
                min_residual_norm = trial_residual_norm
                min_alpha = alpha

            # Check if the trial residual norm is smaller than tolerance
            if trial_residual_norm < self.tol:
                break

            # Backtrack
            alpha *= 0.5

        return min_alpha, line_search_iters

    def optimize(self, u0, **kwargs):
        """
        Main optimization loop
        Args:
            u0: Initial guess
            **kwargs: Additional problem-specific parameters passed to assemble_system
        Returns:
            u: Optimized solution
            success: Boolean indicating convergence
            n_iter: Number of iterations performed
            line_search_iters: Number of line search iterations performed
        """
        u = np.array(u0, dtype=float)
        total_line_search_iters = 0

        for iter in range(self.max_iter):
            # 1. Assemble system
            jacobian = self.assemble_system(u, **kwargs)
            residual = self.calc_residual(u, **kwargs)
            residual_norm = self.residual_norm(residual)

            if self.verbose:
                print(f"Iteration {iter}: residual norm = {residual_norm:.2e}")

            # 2. Check convergence
            if residual_norm < self.tol and iter > 1:
                return u, True, iter, residual_norm, total_line_search_iters

            # 3. Solve for direction
            try:
                direction = self.solve_direction(jacobian, residual) # N_res
            except ValueError as e:
                print(f"Optimization failed: {e}")
                return u, False, iter, residual_norm, total_line_search_iters

            # 4. Line search
            alpha, line_search_iters = self.line_search(u, direction, residual_norm, **kwargs) # iter_line_search * N_res
            total_line_search_iters += line_search_iters
            if self.verbose:
                print(f"Line search: alpha = {alpha:.2e}, iterations = {line_search_iters}")

            # 5. Update solution
            u = u + alpha * direction

        # If we reach here, max iterations exceeded
        return u, False, self.max_iter, residual_norm, total_line_search_iters


class ReactionTerm:
    """
    Class to handle different reaction terms for reaction-diffusion equations.
    
    Supported reaction types:
    - fisher: u(1-u) - Fisher-KPP equation
    - allee: u(1-u)(u-a) - Allee effect with threshold parameter a
    - cubic: u(1-u²) - Allen-Cahn type cubic reaction
    """
    
    def __init__(self, reaction_type="fisher", **params):
        """
        Initialize reaction term.
        
        Args:
            reaction_type (str): Type of reaction term ('fisher', 'allee', 'cubic')
            **params: Additional parameters for specific reaction types
                - For 'allee': 'a' (threshold parameter, default 0.3)
        """
        self.reaction_type = reaction_type.lower()
        self.params = params
        
        # Validate reaction type
        valid_types = ['fisher', 'allee', 'cubic']
        if self.reaction_type not in valid_types:
            raise ValueError(f"Invalid reaction type '{reaction_type}'. Must be one of {valid_types}")
        
        # Set default parameters
        if self.reaction_type == 'allee':
            self.params.setdefault('a', 0.3)
            if not (0 < self.params['a'] < 1):
                raise ValueError("For Allee effect, parameter 'a' must be between 0 and 1")
    
    def evaluate(self, u):
        """
        Evaluate the reaction term f(u).
        
        Args:
            u (numpy.ndarray): Solution vector
            
        Returns:
            numpy.ndarray: Reaction term f(u)
        """
        if self.reaction_type == "fisher":
            return u * (1 - u)
        
        elif self.reaction_type == "allee":
            a = self.params['a']
            return u * (1 - u) * (u - a)
        
        elif self.reaction_type == "cubic":
            return u * (1 - u**2)
        
        else:
            raise ValueError(f"Unknown reaction type: {self.reaction_type}")
    
    def derivative(self, u):
        """
        Evaluate the derivative of the reaction term df/du.
        
        Args:
            u (numpy.ndarray): Solution vector
            
        Returns:
            numpy.ndarray: Derivative df/du
        """
        if self.reaction_type == "fisher":
            return 1 - 2 * u
        
        elif self.reaction_type == "allee":
            a = self.params['a']
            return (1 - u) * (u - a) + u * (1 - u) + u * (u - a)
            # Simplified: 3u² - 2(1+a)u + a
        
        elif self.reaction_type == "cubic":
            return 1 - 3 * u**2
        
        else:
            raise ValueError(f"Unknown reaction type: {self.reaction_type}")
    
    def get_info(self):
        """
        Get information about the current reaction term.
        
        Returns:
            dict: Information about the reaction term
        """
        info = {
            'type': self.reaction_type,
            'formula': self._get_formula(),
            'parameters': self.params.copy()
        }
        return info
    
    def _get_formula(self):
        """Get the mathematical formula for the reaction term."""
        if self.reaction_type == "fisher":
            return "f(u) = u(1-u)"
        elif self.reaction_type == "allee":
            a = self.params['a']
            return f"f(u) = u(1-u)(u-{a})"
        elif self.reaction_type == "cubic":
            return "f(u) = u(1-u²)"
        else:
            return "Unknown"


# Convenience functions for creating reaction terms
def create_fisher_reaction():
    """Create a Fisher-KPP reaction term."""
    return ReactionTerm("fisher")


def create_allee_reaction(a=0.3):
    """
    Create an Allee effect reaction term.
    
    Args:
        a (float): Threshold parameter (0 < a < 1)
    """
    return ReactionTerm("allee", a=a)


def create_cubic_reaction():
    """Create a cubic (Allen-Cahn) reaction term."""
    return ReactionTerm("cubic")


# Example usage and testing
if __name__ == "__main__":
    # Test all reaction terms
    u = np.linspace(0, 1, 11)
    
    print("Testing reaction terms:")
    print("u values:", u)
    print()
    
    # Test Fisher
    fisher = create_fisher_reaction()
    print("Fisher-KPP:", fisher.get_info())
    print("f(u):", fisher.evaluate(u))
    print("df/du:", fisher.derivative(u))
    print()
    
    # Test Allee
    allee = create_allee_reaction(a=0.3)
    print("Allee effect:", allee.get_info())
    print("f(u):", allee.evaluate(u))
    print("df/du:", allee.derivative(u))
    print()
    
    # Test Cubic
    cubic = create_cubic_reaction()
    print("Cubic:", cubic.get_info())
    print("f(u):", cubic.evaluate(u))
    print("df/du:", cubic.derivative(u))


class DiffReac(NewtonOptimizer):
    def __init__(self, tol, max_iter, min_step, verbose, reaction_type="fisher", initial_step_guess=1.0, **reaction_params):
        super().__init__(tol, max_iter, min_step, verbose, initial_step_guess)
        self.reaction_term = ReactionTerm(reaction_type, **reaction_params)

    def calc_laplace(self, u, dx):
        """Calculate laplacian with Dirichlet boundary conditions at left (1) and right ends (0)"""
        n = len(u)

        # Second derivative (u_xx), ignore the left and right boundaries as they are stationary
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2

        return u_xx

    def calc_rhs(self, u, dx):
        """Calculate RHS of PDE: du/dt = u_xx + f(u)"""
        u_xx = self.calc_laplace(u, dx)
        reaction = self.reaction_term.evaluate(u)
        return u_xx + reaction

    def calc_residual(self, u, dt, dx, u_0):
        """
        Calculate residual for fully-implicit scheme
        u: current solution guess
        dt: time step
        dx: spatial step
        u_0: solution from previous time step
        """
        # Calculate RHS terms
        f_u = self.calc_rhs(u, dx)

        # Assemble residual
        # NOTE scale residual by dx**2 for stability
        residual = ((u - u_0) / dt - f_u) * dx**2

        # Set Dirichlet boundary conditions
        residual[0] = 0
        residual[-1] = 0

        return residual

    def assemble_system(self, u, dt, dx, u_0):
        """
        Assemble Jacobian and residual for fully-implicit scheme
        u: current solution guess
        dt: time step
        dx: spatial step
        u_0: solution from previous time step
        """
        n = len(u)

        # Assemble Jacobian
        jacobian = lil_matrix((n, n))

        # Assemble matrix
        # J = I/dt - (laplacian + df/du) = Diag(1/dt + 2/dx² - df/du) + off diag(-1/dx²)
        # NOTE scale both diag off diag by dx**2 for stability
        # Add contribution for diagonal
        reaction_derivative = self.reaction_term.derivative(u)
        diag_main = (1 / dt + 2 - reaction_derivative) * dx**2
        jacobian.setdiag(diag_main)

        # Add contribution for off diag terms
        off_diag_coeffs = -(dx**2)

        # Create indices for off-diagonal elements
        # but ignore the left for left boundary and right for right boundary
        row_indices = np.arange(1, n - 1)
        col_indices_left = (row_indices - 1) % n
        col_indices_right = (row_indices + 1) % n

        # Add contributions for off-diagonal terms
        jacobian[row_indices, col_indices_left] = off_diag_coeffs
        jacobian[row_indices, col_indices_right] = off_diag_coeffs

        return jacobian.tocsr()


class DiffReact1D(SIMULATOR):
    """
    1D Diffusion-Reaction equation solver using fully implicit Newton method.
    Supports different reaction terms: Fisher-KPP, Allee effect, and Cubic (Allen-Cahn).
    
    PDE: ∂u/∂t = ∂²u/∂x² + f(u)
    where f(u) can be:
    - Fisher-KPP: f(u) = u(1-u)
    - Allee effect: f(u) = u(1-u)(u-a)
    - Cubic: f(u) = u(1-u²)
    """

    def __init__(self, verbose, cfg):
        # Physical parameters
        self.L = cfg.L  # Domain length
        self.reaction_type = cfg.reaction_type  # Type of reaction term
        
        # Reaction term parameters
        self.reaction_params = {}
        if hasattr(cfg, 'allee_threshold'):
            self.reaction_params['a'] = cfg.allee_threshold
        
        # Numerical parameters
        self.n_space = cfg.n_space  # Number of grid points
        self.dx = self.L / self.n_space
        self.cfl = cfg.cfl  # CFL number for timestep calculation
        
        # Newton solver parameters
        self.tol = cfg.tol  # Convergence tolerance
        self.max_iter = cfg.max_iter  # Maximum Newton iterations
        self.min_step = cfg.min_step  # Minimum step size for line search
        self.initial_step_guess = cfg.initial_step_guess  # Initial step size for line search
        
        # Create spatial grid
        self.x = np.linspace(0, self.L, self.n_space)
        
        # Initialize solution with step function initial condition
        self.u = self.initialize_condition()

        # Cost tracking variables
        self.total_newton_iters = 0
        self.total_line_search_iters = 0

        self.solver = DiffReac(
            tol=self.tol, 
            max_iter=self.max_iter, 
            min_step=self.min_step, 
            verbose=False,
            reaction_type=self.reaction_type,
            initial_step_guess=self.initial_step_guess,
            **self.reaction_params
        )
        
        # Output directory
        self.dump_dir = cfg.dump_dir + f"_nspace{self.n_space}_cfl{format_param_for_path(self.cfl)}_tol{format_param_for_path(self.tol)}_minstep{format_param_for_path(self.min_step)}_initstep{format_param_for_path(self.initial_step_guess)}"
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)
        
        # Base initialization
        super().__init__(verbose, cfg)

    def initialize_condition(self):
        """Initialize with step function initial condition matching the original solver"""
        # Step function: u=1 for 0 <= x <= 2, u=0 elsewhere
        # This matches the original: u_0 = np.where((x >= 0) & (x <= 2), 1.0, 0.0)
        u = np.zeros_like(self.x)
        u[(self.x >= 0) & (self.x <= 2)] = 1.0
        return u

    def cal_dt(self):
        """
        Calculate timestep for diffusion-reaction equation using CFL condition.
        For diffusion-reaction: dt = cfl * dx² / D where D=1 (diffusion coefficient)
        """
        # CFL-based timestep for diffusion: dt = cfl * dx² / D
        # where D = 1 is the diffusion coefficient in our equation
        dt_diffusion = self.cfl * self.dx**2
        
        # For reaction terms, we also need to consider the reaction timescale
        # For Fisher: reaction timescale ~ 1, so we use a conservative factor
        dt_reaction = 0.1  # Conservative choice
        
        # Use the more restrictive timestep
        dt = min(dt_diffusion, dt_reaction)
        
        # Ensure we don't exceed the recording interval
        dt = min(dt, self.record_dt / 10)
        
        return dt

    def step(self, dt):
        """
        Perform a single time step using fully implicit Newton method.
        """
        # Use current solution as initial guess for next time step
        u_guess = self.u.copy()
        
        # Solve for next time step using Newton method
        u_next, success, iters, residual_norm, line_search_iters = self.solver.optimize(
            u_guess, dt=dt, dx=self.dx, u_0=self.u
        )
        
        # Accumulate cost tracking
        self.total_newton_iters += iters
        self.total_line_search_iters += line_search_iters
        
        if not success:
            if self.verbose:
                print(f"Warning: Newton solver failed to converge at time {self.current_time:.6f}")
                print(f"  Iterations: {iters}, Residual norm: {residual_norm:.2e}")
            # Use the best available solution
            self.u = u_next
        else:
            self.u = u_next
            
        if self.verbose and iters > 10:
            print(f"  Newton iterations: {iters}, Residual: {residual_norm:.2e}")

    def dump(self):
        """Save current state including data file and visualization"""
        # Create filename base
        file_base = os.path.join(self.dump_dir, f"res_{self.record_frame}")
        
        # Save HDF5 data file
        with h5py.File(f"{file_base}.h5", "w") as f:
            f.create_dataset("x", data=self.x)
            f.create_dataset("u", data=self.u)
            f.create_dataset("time", data=self.current_time)
            f.create_dataset("reaction_type", data=self.reaction_type.encode('utf-8'))
            
            # Save reaction term info
            reaction_info = self.solver.reaction_term.get_info()
            f.create_dataset("reaction_formula", data=reaction_info['formula'].encode('utf-8'))
            if reaction_info['parameters']:
                for key, value in reaction_info['parameters'].items():
                    f.create_dataset(f"reaction_param_{key}", data=value)
        
        # Create and save plot
        plt.figure(figsize=(12, 8))
        
        # Main solution plot
        plt.subplot(2, 1, 1)
        plt.plot(self.x, self.u, "b-", linewidth=2, label="Solution")
        plt.xlabel("Position (x)")
        plt.ylabel("Solution (u)")
        plt.title(f"Diffusion-Reaction Equation ({self.reaction_type.upper()}) - Time = {self.current_time:.3f}")
        plt.grid(True)
        plt.legend()
        
        # Reaction term plot
        plt.subplot(2, 1, 2)
        reaction = self.solver.reaction_term.evaluate(self.u)
        plt.plot(self.x, reaction, "r-", linewidth=2, label=f"Reaction term: {reaction_info['formula']}")
        plt.xlabel("Position (x)")
        plt.ylabel("Reaction term f(u)")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{file_base}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def post_process(self):
        """Post-processing: save metadata and final statistics"""
        # Calculate cost as: sum_n_newton_iter * N_res + sum_iter_line_search * N_res
        # where N_res is the number of residual calculations per iteration (n_space for 1D)
        newton_cost = 3 * self.total_newton_iters * self.n_space
        line_search_cost = self.total_line_search_iters * self.n_space
        cost = newton_cost + line_search_cost
        
        # Calculate some statistics
        final_max = np.max(self.u)
        final_min = np.min(self.u)
        
        # Find wave front position (where u = 0.5)
        try:
            wave_front_idx = np.argmin(np.abs(self.u - 0.5))
            wave_front_pos = self.x[wave_front_idx]
        except:
            wave_front_pos = None
        
        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            meta = {
                "cost": cost,
                "newton_cost": newton_cost,
                "line_search_cost": line_search_cost,
                "total_newton_iters": int(self.total_newton_iters),
                "total_line_search_iters": int(self.total_line_search_iters),
                "reaction_type": self.reaction_type,
                "reaction_params": self.reaction_params,
                "n_space": int(self.n_space),
                "dx": float(self.dx),
                "total_steps": int(self.num_steps),
                "final_time": float(self.current_time),
                "final_max": float(final_max),
                "final_min": float(final_min),
                "wave_front_position": float(wave_front_pos) if wave_front_pos is not None else None,
                "tol": format_param_for_path(self.tol),
                "max_iter": int(self.max_iter),
                "min_step": format_param_for_path(self.min_step),
                "initial_step_guess": format_param_for_path(self.initial_step_guess),
                "cfl": format_param_for_path(self.cfl),
            }
            json.dump(meta, f, indent=4)
        
        if self.verbose:
            print(f"Run cost: {cost}")
            print(f"  Newton cost: {newton_cost} (iterations: {self.total_newton_iters})")
            print(f"  Line search cost: {line_search_cost} (iterations: {self.total_line_search_iters})")
            print(f"Final statistics:")
            print(f"  Max value: {final_max:.4f}")
            print(f"  Min value: {final_min:.4f}")
            if wave_front_pos is not None:
                print(f"  Wave front position: {wave_front_pos:.4f}")
