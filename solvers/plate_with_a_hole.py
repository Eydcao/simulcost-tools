import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import json
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from .base_solver import SIMULATOR


class PlateWithHole(SIMULATOR):
    """
    2D finite element analysis of a plate with a circular hole under uniaxial tension.
    Solves the plane strain elasticity problem using 4-node quadrilateral elements.
    """

    def __init__(self, verbose, cfg):
        # Physical parameters
        self.L = cfg.L  # Half width (quarter model)
        self.H = cfg.H  # Half height
        self.R = cfg.R  # Hole radius

        # Material properties
        self.E = cfg.E  # Young's modulus
        self.nu = cfg.nu  # Poisson's ratio
        self.traction = cfg.traction  # Applied traction

        # Mesh parameters
        self.nx = cfg.nx  # Number of elements in x
        self.ny = cfg.ny  # Number of elements in y
        self.hx = self.L / self.nx
        self.hy = self.H / self.ny

        # Volume for error calculation
        self.V = self.hx * self.hy / 4.0

        # Output directory
        self.dump_dir = cfg.dump_dir + f"_nx_{self.nx}_ny_{self.ny}"
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Initialize mesh and solution
        self.coords, self.elements, self.activated_nodes = self.generate_mesh()
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
        """Generate structured mesh and remove elements inside hole"""
        x = np.linspace(0, self.L, self.nx + 1)
        y = np.linspace(0, self.H, self.ny + 1)
        xv, yv = np.meshgrid(x, y)
        coords = np.column_stack([xv.flatten(), yv.flatten()])
        elements = []

        node_id = lambda i, j: j * (self.nx + 1) + i
        n_nodes = coords.shape[0]
        activated_nodes = np.zeros(n_nodes)

        for j in range(self.ny):
            for i in range(self.nx):
                n1 = node_id(i, j)
                n2 = node_id(i + 1, j)
                n3 = node_id(i + 1, j + 1)
                n4 = node_id(i, j + 1)

                # Check if element centroid is outside hole
                xc = np.mean([coords[n1, 0], coords[n2, 0], coords[n3, 0], coords[n4, 0]])
                yc = np.mean([coords[n1, 1], coords[n2, 1], coords[n3, 1], coords[n4, 1]])

                if np.sqrt(xc**2 + yc**2) >= self.R:
                    elements.append([n1, n2, n3, n4])
                    activated_nodes[[n1, n2, n3, n4]] = 1.0

        return coords, np.array(elements), activated_nodes

    def shape_functions(self, xi, eta):
        """4-node quadrilateral shape functions and derivatives"""
        N = 0.25 * np.array([
            (1 - xi)*(1 - eta),
            (1 + xi)*(1 - eta),
            (1 + xi)*(1 + eta),
            (1 - xi)*(1 + eta)
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

            for xi, eta in gauss_points:
                N, dN_dxi = self.shape_functions(xi, eta)
                J = xe.T @ dN_dxi
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

            # Assembly
            dofs = np.array([[2*node, 2*node+1] for node in element]).flatten()
            for i in range(8):
                for j in range(8):
                    K[dofs[i], dofs[j]] += ke[i, j]

        return csr_matrix(K), f

    def apply_boundary_conditions(self, K, f):
        """Apply Neumann and Dirichlet boundary conditions"""
        tol = 1e-8

        # Apply traction on right edge
        right_nodes = np.where(np.abs(self.coords[:, 0] - self.L) < tol)[0]
        right_nodes = sorted(right_nodes, key=lambda i: self.coords[i, 1])

        # Apply edge traction
        for i in range(len(right_nodes) - 1):
            n1, n2 = right_nodes[i], right_nodes[i + 1]
            x1, y1 = self.coords[n1]
            x2, y2 = self.coords[n2]
            length_edge = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            traction_vec = np.array([self.traction, 0])
            fe = np.zeros(4)
            fe[0:2] = traction_vec * length_edge / 2
            fe[2:4] = traction_vec * length_edge / 2
            dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            for j in range(4):
                f[dofs[j]] += fe[j]

        # Apply Dirichlet BCs (symmetry)
        bc_dofs = []
        for i in range(self.n_nodes):
            x, y = self.coords[i]
            if np.abs(x) < tol:
                bc_dofs.append(2*i)     # ux = 0
            if np.abs(y) < tol:
                bc_dofs.append(2*i + 1) # uy = 0

            # Deactivated nodes (inside hole)
            if self.activated_nodes[i] < 0.5:
                bc_dofs.append(2*i)
                bc_dofs.append(2*i + 1)

        bc_dofs = np.array(bc_dofs)
        free_dofs = np.setdiff1d(np.arange(2 * self.n_nodes), bc_dofs)

        return free_dofs, bc_dofs

    def solve(self):
        """Solve the linear system"""
        K, f = self.assemble_system()
        free_dofs, bc_dofs = self.apply_boundary_conditions(K, f)

        u = np.zeros(2 * self.n_nodes)
        u[free_dofs] = spsolve(K[free_dofs][:, free_dofs], f[free_dofs])

        return u

    def calculate_stress_error(self, u):
        """Calculate normalized stress error along x=0"""
        sigma_data = []
        gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)),
                        ( 1/np.sqrt(3), -1/np.sqrt(3)),
                        ( 1/np.sqrt(3),  1/np.sqrt(3)),
                        (-1/np.sqrt(3),  1/np.sqrt(3))]

        for e in range(len(self.elements)):
            element = self.elements[e]
            xe = self.coords[element]
            ue = np.array([u[2*n + i] for n in element for i in range(2)])

            for xi, eta in gauss_points:
                N, dN_dxi = self.shape_functions(xi, eta)
                J = xe.T @ dN_dxi
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
                sigma_xx = stress[0]

                # Physical coordinates of Gauss point
                x_gp = N @ xe[:, 0]
                y_gp = N @ xe[:, 1]

                # Check if on centerline and outside hole
                if abs(x_gp) < self.hx/2 and y_gp > self.R + 1e-6:
                    # Analytical solution for infinite plate with hole
                    sigma_xx_ana = self.traction * (1 + self.R**2/2/y_gp**2 + 3*self.R**4/2/y_gp**4)
                    sigma_data.append([y_gp, sigma_xx, sigma_xx_ana])

        if len(sigma_data) == 0:
            return 0.0

        sigma_data = np.array(sigma_data)
        sigma_data = sigma_data[sigma_data[:, 0].argsort()]

        sigma_fem = sigma_data[:, 1]
        sigma_ana = sigma_data[:, 2]

        # Calculate normalized error
        error_numerator = np.sqrt(np.sum((sigma_fem - sigma_ana)**2)) * self.V
        error_denominator = self.traction * self.V * len(sigma_fem)

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
                'traction': self.traction,
                'R': self.R,
                'L': self.L,
                'H': self.H
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
        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            meta = {
                "cost": cost,
                "nx": int(self.nx),
                "ny": int(self.ny),
                "n_elements": len(self.elements),
                "n_nodes": self.n_nodes,
                "n_dof": 2 * self.n_nodes
            }
            json.dump(meta, f, indent=4)
        if self.verbose:
            print(f"Run cost: {cost}")