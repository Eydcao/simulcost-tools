import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class NewtonOptimizer:
    def __init__(
        self,
        tol,
        max_iter,
        min_step,
        verbose,
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.min_step = min_step
        self.verbose = verbose

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

    def solve_direction(self, jacobian, residual):
        """
        Solve for optimization direction using factorization
        """
        try:
            if isinstance(jacobian, np.ndarray):
                direction = -linalg.solve(jacobian, residual)
            elif isinstance(jacobian, csr_matrix):
                direction = -spsolve(jacobian, residual)
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

    def line_search(self, u, direction, initial_residual_norm, **kwargs):
        """
        Perform line search with greedy approach
        """
        alpha = 1.0
        current_u = u
        min_alpha = alpha
        min_residual_norm = float("inf")

        while alpha > self.min_step:
            # Try step
            trial_u = current_u + alpha * direction
            trial_residual = self.calc_residual(trial_u, **kwargs)
            trial_residual_norm = self.residual_norm(trial_residual)

            # Track minimum residual norm and associated alpha
            if trial_residual_norm < min_residual_norm:
                min_residual_norm = trial_residual_norm
                min_alpha = alpha

            # Check if the trial residual norm is smaller than tolerance
            if trial_residual_norm < self.tol:
                break

            # Backtrack
            alpha *= 0.5

        return min_alpha

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
        """
        u = np.array(u0, dtype=float)

        for iter in range(self.max_iter):
            # 1. Assemble system
            jacobian = self.assemble_system(u, **kwargs)
            residual = self.calc_residual(u, **kwargs)
            residual_norm = self.residual_norm(residual)

            if self.verbose:
                print(f"Iteration {iter}: residual norm = {residual_norm:.2e}")

            # 2. Check convergence
            if residual_norm < self.tol and iter > 1:
                return u, True, iter, residual_norm

            # 3. Solve for direction
            try:
                direction = self.solve_direction(jacobian, residual)
            except ValueError as e:
                print(f"Optimization failed: {e}")
                return u, False, iter, residual_norm

            # 4. Line search
            alpha = self.line_search(u, direction, residual_norm, **kwargs)
            if self.verbose:
                print(f"Line search: alpha = {alpha:.2e}")

            # 5. Update solution
            u = u + alpha * direction

        # If we reach here, max iterations exceeded
        return u, False, self.max_iter, residual_norm
