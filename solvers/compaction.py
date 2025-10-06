import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import json
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from .base_solver import SIMULATOR


class Compaction(SIMULATOR):
    """
    2D finite element analysis of bar compaction under self-weight.
    Solves the plane strain elasticity problem with gravity loading.
    """

    def __init__(self, verbose, cfg):
        # Physical parameters
        self.length = cfg.length  # Bar length
        self.height = cfg.height  # Bar height

        # Material properties
        self.E = cfg.E  # Young's modulus
        self.nu = cfg.nu  # Poisson's ratio
        self.rho = cfg.rho  # Density
        self.g = cfg.g  # Gravity

        # Mesh parameters
        self.nx = cfg.nx  # Number of elements in x
        self.ny = cfg.ny  # Number of elements in y
        self.hx = self.length / self.nx
        self.hy = self.height / self.ny

        # Volume for error calculation
        self.V = self.hx * self.hy / 4.0

        # Body force
        self.body_force = np.array([0, -self.rho * self.g])

        # Output directory
        self.dump_dir = cfg.dump_dir + f"_nx_{self.nx}_ny_{self.ny}"
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Initialize mesh and solution
        self.coords, self.elements = self.generate_mesh()
        self.n_nodes = self.coords.shape[0]

        # Material matrix for plane strain
        self.D = self.plane_strain_D(self.E, self.nu)

        # Base initialization
        super().__init__(verbose, cfg)

    def plane_strain_D(self, E, nu):
        """Material matrix for plane strain conditions"""
        coeff = E / ((1 + nu) * (1 - 2 * nu))
        return coeff * np.array([
            [1 - nu,     nu,       0],
            [nu,     1 - nu,       0],
            [0,          0, (1 - 2 * nu) / 2]
        ])

    def generate_mesh(self):
        """Generate structured rectangular mesh"""
        x = np.linspace(0, self.length, self.nx + 1)
        y = np.linspace(0, self.height, self.ny + 1)
        xv, yv = np.meshgrid(x, y)
        coords = np.column_stack([xv.flatten(), yv.flatten()])

        elements = []
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = j * (self.nx + 1) + i
                n2 = n1 + 1
                n3 = n2 + (self.nx + 1)
                n4 = n1 + (self.nx + 1)
                elements.append([n1, n2, n3, n4])

        return coords, np.array(elements)

    def shape_functions(self, xi, eta):
        """4-node quadrilateral shape functions and derivatives"""
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta)
        ])
        dN_dxi = 0.25 * np.array([
            [-(1 - eta), -(1 - xi)],
            [ (1 - eta), -(1 + xi)],
            [ (1 + eta),  (1 + xi)],
            [-(1 + eta),  (1 - xi)]
        ])
        return N, dN_dxi

    def assemble_system(self):
        """Assemble global stiffness matrix and force vector"""
        K = lil_matrix((2 * self.n_nodes, 2 * self.n_nodes))
        f = np.zeros(2 * self.n_nodes)

        # Gauss points for 2x2 integration
        gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)),
                        ( 1/np.sqrt(3), -1/np.sqrt(3)),
                        ( 1/np.sqrt(3),  1/np.sqrt(3)),
                        (-1/np.sqrt(3),  1/np.sqrt(3))]

        # Element assembly
        for e in range(len(self.elements)):
            element = self.elements[e]
            xe = self.coords[element]
            ke = np.zeros((8, 8))
            fe = np.zeros(8)

            for xi, eta in gauss_points:
                N, dN_dxi = self.shape_functions(xi, eta)
                J = dN_dxi.T @ xe
                detJ = np.linalg.det(J)
                dN_dx = np.linalg.solve(J, dN_dxi.T).T

                # Strain-displacement matrix
                B = np.zeros((3, 8))
                for i in range(4):
                    B[0, 2*i]     = dN_dx[i, 0]
                    B[1, 2*i+1]   = dN_dx[i, 1]
                    B[2, 2*i]     = dN_dx[i, 1]
                    B[2, 2*i+1]   = dN_dx[i, 0]

                ke += B.T @ self.D @ B * detJ

                # Body force contribution
                for i in range(4):
                    fe[2*i:2*i+2] += N[i] * self.body_force * detJ

            # Assembly
            dofs = np.array([[2*node, 2*node+1] for node in element]).flatten()
            for i in range(8):
                for j in range(8):
                    K[dofs[i], dofs[j]] += ke[i, j]
            f[dofs] += fe

        return csr_matrix(K), f

    def apply_boundary_conditions(self, K, f):
        """Apply boundary conditions (bottom edge fixed)"""
        tol = 1e-8
        fixed_nodes = np.where(self.coords[:, 1] < tol)[0]
        fixed_dofs = np.array([[2*n, 2*n+1] for n in fixed_nodes]).flatten()
        free_dofs = np.setdiff1d(np.arange(2 * self.n_nodes), fixed_dofs)

        return free_dofs, fixed_dofs

    def solve(self):
        """Solve the linear system"""
        K, f = self.assemble_system()
        free_dofs, fixed_dofs = self.apply_boundary_conditions(K, f)

        u = np.zeros(2 * self.n_nodes)
        u[free_dofs] = spsolve(K[free_dofs][:, free_dofs], f[free_dofs])

        return u

    def calculate_stress_error(self, u):
        """Calculate normalized stress error at Gauss points"""
        # Gauss points for stress evaluation
        gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)),
                        ( 1/np.sqrt(3), -1/np.sqrt(3)),
                        ( 1/np.sqrt(3),  1/np.sqrt(3)),
                        (-1/np.sqrt(3),  1/np.sqrt(3))]

        stress_data = []

        for e in range(len(self.elements)):
            element = self.elements[e]
            xe = self.coords[element]
            ue = np.array([u[2*n + i] for n in element for i in range(2)])

            for xi, eta in gauss_points:
                N, dN_dxi = self.shape_functions(xi, eta)
                J = dN_dxi.T @ xe
                dN_dx = np.linalg.solve(J, dN_dxi.T).T

                # Strain-displacement matrix
                B = np.zeros((3, 8))
                for i in range(4):
                    B[0, 2*i]     = dN_dx[i, 0]
                    B[1, 2*i+1]   = dN_dx[i, 1]
                    B[2, 2*i]     = dN_dx[i, 1]
                    B[2, 2*i+1]   = dN_dx[i, 0]

                strain = B @ ue
                stress = self.D @ strain
                sigma_yy = stress[1]

                # Physical coordinates of Gauss point
                x_gp = N @ xe[:, 0]
                y_gp = N @ xe[:, 1]

                # Analytical solution for vertical stress
                sigma_yy_analytical = self.rho * self.g * (y_gp - self.height)
                stress_data.append([y_gp, sigma_yy, sigma_yy_analytical])

        if len(stress_data) == 0:
            return 0.0

        stress_data = np.array(stress_data)
        stress_data = stress_data[stress_data[:, 0].argsort()]

        sigma_fem = stress_data[:, 1]
        sigma_ana = stress_data[:, 2]

        # Calculate normalized error
        error_numerator = np.sqrt(np.sum((sigma_fem - sigma_ana)**2)) * self.V
        error_denominator = self.g * self.rho * self.height * self.V * len(sigma_fem)

        return error_numerator / error_denominator

    def pre_process(self):
        """Initialize simulation"""
        pass

    def cal_dt(self):
        """Not applicable for static problems"""
        return 1.0

    def call_back(self):
        """Called after recording"""
        pass

    def dump(self):
        """Save simulation results"""
        # Solve the system
        u = self.solve()

        # Calculate error metric
        error = self.calculate_stress_error(u)

        # Save results
        results = {
            'displacement': u.tolist(),
            'coordinates': self.coords.tolist(),
            'elements': self.elements.tolist(),
            'error': error,
            'parameters': {
                'nx': self.nx,
                'ny': self.ny,
                'E': self.E,
                'nu': self.nu,
                'rho': self.rho,
                'g': self.g,
                'length': self.length,
                'height': self.height
            }
        }

        # Save to HDF5
        output_file = os.path.join(self.dump_dir, f"frame_{self.record_frame:04d}.h5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('displacement', data=u)
            f.create_dataset('coordinates', data=self.coords)
            f.create_dataset('elements', data=self.elements)
            f.attrs['error'] = error
            f.attrs['nx'] = self.nx
            f.attrs['ny'] = self.ny

        # Save to JSON
        json_file = os.path.join(self.dump_dir, f"frame_{self.record_frame:04d}.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"Frame {self.record_frame}: Error = {error:.6e}")

        self.record_frame += 1

    def estimate_cost(self):
        """Estimate computational cost based on problem size"""
        n_dof = 2 * self.n_nodes
        n_elements = len(self.elements)

        # Rough FLOP estimate: assembly + solve
        assembly_flops = n_elements * 8**3 * 4  # 4 Gauss points, 8x8 matrix ops
        solve_flops = n_dof**2  # Rough estimate for sparse solve

        return assembly_flops + solve_flops

    def post_process(self):
        """Post-processing: save metadata"""
        cost = self.estimate_cost()
        
        # Calculate the error from the last dump
        u = self.solve()
        error = self.calculate_stress_error(u)
        
        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            meta = {
                "cost": cost,
                "nx": int(self.nx),
                "ny": int(self.ny),
                "n_elements": len(self.elements),
                "n_nodes": self.n_nodes,
                "n_dof": 2 * self.n_nodes,
                "error": error
            }
            json.dump(meta, f, indent=4)
        if self.verbose:
            print(f"Run cost: {cost}")
            print(f"Stress error: {error:.6e}")