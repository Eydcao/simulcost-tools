import numpy as np
from base import NewtonOptimizer
from scipy.sparse import diags, lil_matrix


class DiffReac(NewtonOptimizer):
    def __init__(self, tol, max_iter, min_step, verbose):
        super().__init__(tol, max_iter, min_step, verbose)

    def calc_laplace(self, u, dx):
        """Calculate laplacian with Dirichlet boundary conditions at left (1) and right ends (0)"""
        n = len(u)

        # Second derivative (u_xx), ignore the left and right boundaries as they are stationary
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2

        return u_xx

    def calc_rhs(self, u, dx):
        """Calculate RHS of PDE: du/dt = u_xx + u(1-u)"""
        u_xx = self.calc_laplace(u, dx)
        return u_xx + u * (1 - u)

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
        # J = I/dt - (laplacian + (1-2u)) = Diag(1/dt + 2 - (1 - 2u)) + off diag(-1) = Diag(1/dt + 1 + 2u)) + off diag(-1)

        # NOTE scale both diag off diag by dx**2 for stability
        # Add contribution for diagonal
        diag_main = (1 / dt + 1 + 2 * u) * dx**2
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
