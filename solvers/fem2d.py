import taichi as ti
import numpy as np

# scipy only used for mesh generation, not for linear solve
import scipy.sparse  # Only for mesh generation
import meshio
import os
import matplotlib.pyplot as plt

from .base_solver import SIMULATOR
from .fastipc_utils.common.physics.fixed_corotated import *
from .fastipc_utils.common.math.math_tools import *


@ti.data_oriented
class FEM2D(SIMULATOR):
    def __init__(self, verbose, cfg):
        super().__init__(verbose, cfg)

        self.sparsity_pattern_analyzed = False

        self.dim = cfg.dim
        self.E = cfg.E
        self.nu = cfg.nu
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
        print(f"Lamé parameters: λ={self.la}, μ={self.mu}")
        self.density = cfg.density
        self.cfl = cfg.cfl
        self.dx = cfg.dx

        # Case type and gravity
        self.case = cfg.get("case", "cantilever")
        self.gravity = cfg.envs_params.get("gravity", -9.8)

        # Newton solver parameters
        # newton_v_res_tol: velocity-like residual tolerance
        # Convergence criterion: |Δx| / dt < newton_v_res_tol
        # where Δx is the position correction from Newton solver
        self.max_newton_iter = cfg.max_newton_iter
        self.newton_v_res_tol = cfg.envs_params.newton_v_res_tol

        mesh = self._get_mesh(self.dx, cfg)
        self.mesh_particles = mesh.points
        self.mesh_elements = mesh.cells[0].data
        self.mesh_scale = cfg.mesh_scale
        self.mesh_offset = cfg.mesh_offset

        self.n_particles = len(self.mesh_particles)
        self.n_elements = len(self.mesh_elements)

        self.real = ti.f64
        ti.init(arch=ti.cpu, default_fp=self.real, verbose=self.verbose, cpu_max_num_threads=8, debug=False)

        self.cnt = ti.field(dtype=ti.i32, shape=())
        self.x = ti.Vector.field(self.dim, dtype=self.real)
        self.xPrev = ti.Vector.field(self.dim, dtype=self.real)
        self.xTilde = ti.Vector.field(self.dim, dtype=self.real)
        self.xn = ti.Vector.field(self.dim, dtype=self.real)
        self.v = ti.Vector.field(self.dim, dtype=self.real)
        self.m = ti.field(dtype=self.real)
        self.zero = ti.Vector.field(self.dim, dtype=self.real)
        self.restT = ti.Matrix.field(self.dim, self.dim, dtype=self.real)
        self.vertices = ti.field(dtype=ti.i32)

        ti.root.dense(ti.k, self.n_particles).place(self.x, self.xPrev, self.xTilde, self.xn, self.v, self.m)
        ti.root.dense(ti.k, self.n_particles).place(self.zero)
        ti.root.dense(ti.i, self.n_elements).place(self.restT)
        ti.root.dense(ti.ij, (self.n_elements, self.dim + 1)).place(self.vertices)

        # Calculate required sparse matrix storage size
        # Inertia: n_particles * dim diagonal entries
        # Elasticity: n_elements * (dim+1)*dim * (dim+1)*dim entries per element
        inertia_entries = self.n_particles * self.dim
        elasticity_entries = self.n_elements * (self.dim + 1) * self.dim * (self.dim + 1) * self.dim
        max_matrix_entries = int((inertia_entries + elasticity_entries) * 1.2)  # 20% safety margin

        self.data_rhs = ti.field(self.real, shape=self.n_particles * self.dim)
        self.data_mat = ti.field(self.real, shape=(3, max_matrix_entries))
        self.data_sol = ti.field(self.real, shape=self.n_particles * self.dim)

        self.dump_dir = cfg.dump_dir
        self.fixed_nodes = []
        self.bc_initialized = False  # Flag to track if BCs have been set up
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

        # Initialize Taichi sparse linear solver
        # Using LLT (Cholesky) for symmetric positive definite systems
        # LDLT is also good for symmetric indefinite, LU for general
        self.solver = ti.linalg.SparseSolver(solver_type="LLT")
        self.n_dof = self.n_particles * self.dim

        # Taichi field for boundary condition DOFs
        # Store list of DOF indices that have Dirichlet boundary conditions
        max_bc_dofs = self.n_dof  # Conservative upper bound
        self.bc_dofs = ti.field(dtype=ti.i32, shape=max_bc_dofs)
        self.num_bc_dofs = ti.field(dtype=ti.i32, shape=())

        # Boolean lookup array for fast BC checking (O(1) instead of O(n))
        self.is_bc_dof = ti.field(dtype=ti.i32, shape=self.n_dof)

        # Cost counters for tracking computational work
        self.cnt_hessian_gradient = 0  # Counts hessian+gradient assembly (cost: n_elements each)
        self.cnt_energy = 0  # Counts energy evaluation (cost: n_elements each)
        self.cnt_linear_solve = 0  # Counts sparse linear solve (cost: (n_particles*dim)^2 each)

        # Energy tracking for convergence analysis
        self.energy_log_kin = []
        self.energy_log_pot = []
        self.energy_log_gra = []
        self.energy_log_tot = []

    def _get_mesh(self, dx, cfg):
        mesh_path = f"output/mesh/{self.case}_dx{dx}.obj"
        if os.path.exists(mesh_path):
            return meshio.read(mesh_path)

        Lx = cfg.envs_params.Lx
        Ly = cfg.envs_params.Ly
        n_gx_e = int(Lx / dx)
        n_gy_e = int(n_gx_e * Ly / Lx)
        dy = Ly / n_gy_e

        # Get starting position based on case
        if self.case == "vibration_bar":
            x_start = cfg.envs_params.get("x_start", 0.0)
            y_start = cfg.envs_params.get("y_start", 0.0)
        elif self.case == "twisting_column":
            x_start = 0.0
            y_start = 0.0
        else:  # cantilever
            x_start = 0.0
            y_start = 5.0

        n_vertices = (n_gx_e + 1) * (n_gy_e + 1)
        v_pos = np.zeros((n_vertices, 2), dtype=float)
        for i in range(n_gx_e + 1):
            for j in range(n_gy_e + 1):
                v_pos[i * (n_gy_e + 1) + j] = np.array([x_start + dx * i, y_start + dy * j])

        from scipy.spatial import Delaunay

        tri = Delaunay(v_pos)
        cells = [("triangle", tri.simplices)]

        # Add a z-coordinate of 0 to make the points 3D
        points_3d = np.hstack([v_pos, np.zeros((v_pos.shape[0], 1))])

        mesh = meshio.Mesh(points_3d, cells)
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
        mesh.write(mesh_path)
        return mesh

    @ti.func
    def compute_T(self, i):
        if ti.static(self.dim == 2):
            ab = self.x[self.vertices[i, 1]] - self.x[self.vertices[i, 0]]
            ac = self.x[self.vertices[i, 2]] - self.x[self.vertices[i, 0]]
            return ti.Matrix([[ab[0], ac[0]], [ab[1], ac[1]]])
        else:
            ab = self.x[self.vertices[i, 1]] - self.x[self.vertices[i, 0]]
            ac = self.x[self.vertices[i, 2]] - self.x[self.vertices[i, 0]]
            ad = self.x[self.vertices[i, 3]] - self.x[self.vertices[i, 0]]
            return ti.Matrix([[ab[0], ac[0], ad[0]], [ab[1], ac[1], ad[1]], [ab[2], ac[2], ad[2]]])

    @ti.kernel
    def compute_restT_and_m(self):
        for i in range(self.n_elements):
            self.restT[i] = self.compute_T(i)
            mass = self.restT[i].determinant() / self.dim / (self.dim - 1) * self.density / (self.dim + 1)
            if mass < 0.0:
                print("FATAL ERROR : mesh inverted")
            for d in ti.static(range(self.dim + 1)):
                self.m[self.vertices[i, d]] += mass

    @ti.kernel
    def compute_xn_and_xTilde(self, dt: ti.f64):
        for i in range(self.n_particles):
            self.xn[i] = self.x[i]
            self.xTilde[i] = self.x[i] + dt * self.v[i]
            # Apply gravity (gravity is negative for downward)
            self.xTilde[i][1] += dt * dt * self.gravity

    @ti.kernel
    def compute_energy(self, dt: ti.f64) -> ti.f64:
        total_energy = 0.0
        # inertia
        for i in range(self.n_particles):
            total_energy += 0.5 * self.m[i] * (self.x[i] - self.xTilde[i]).norm_sqr()
        # elasticity
        for e in range(self.n_elements):
            F = self.compute_T(e) @ self.restT[e].inverse()
            vol0 = self.restT[e].determinant() / self.dim / (self.dim - 1)
            U, sig, V = svd(F)
            if ti.static(self.dim == 2):
                total_energy += elasticity_energy(ti.Vector([sig[0, 0], sig[1, 1]]), self.la, self.mu) * dt * dt * vol0
            else:
                total_energy += (
                    elasticity_energy(ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]]), self.la, self.mu) * dt * dt * vol0
                )
        return total_energy

    @ti.kernel
    def compute_kinetic_energy(self) -> ti.f64:
        """Compute kinetic energy: 0.5 * m * v^2"""
        kin_energy = 0.0
        for i in range(self.n_particles):
            kin_energy += 0.5 * self.m[i] * self.v[i].norm_sqr()
        return kin_energy

    @ti.kernel
    def compute_elastic_potential_energy(self) -> ti.f64:
        """Compute elastic potential energy"""
        pot_energy = 0.0
        for e in range(self.n_elements):
            F = self.compute_T(e) @ self.restT[e].inverse()
            vol0 = self.restT[e].determinant() / self.dim / (self.dim - 1)
            U, sig, V = svd(F)
            if ti.static(self.dim == 2):
                pot_energy += elasticity_energy(ti.Vector([sig[0, 0], sig[1, 1]]), self.la, self.mu) * vol0
            else:
                pot_energy += elasticity_energy(ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]]), self.la, self.mu) * vol0
        return pot_energy

    @ti.kernel
    def compute_gravitational_energy(self, y_ref: ti.f64) -> ti.f64:
        """Compute gravitational potential energy relative to reference height

        E_grav = m * g * (y - y_ref)

        Args:
            y_ref: Reference height (typically initial COM height)

        Returns:
            Gravitational potential energy
        """
        gra_energy = 0.0
        for i in range(self.n_particles):
            # Gravitational potential energy: m * g * height
            # y[1] is the vertical coordinate
            # NOTE gravity is negative for downward direction
            gra_energy -= self.m[i] * self.gravity * (self.x[i][1] - y_ref)
        return gra_energy

    def log_energies(self):
        """Compute and log kinetic, potential, gravitational, and total energies"""
        kin = self.compute_kinetic_energy()
        pot = self.compute_elastic_potential_energy()
        gra = self.compute_gravitational_energy(self.y_ref)
        tot = kin + pot + gra
        self.energy_log_kin.append(kin)
        self.energy_log_pot.append(pot)
        self.energy_log_gra.append(gra)
        self.energy_log_tot.append(tot)

    @ti.kernel
    def compute_hessian_and_gradient(self, dt: ti.f64):
        self.cnt[None] = 0
        # inertia
        for i in range(self.n_particles):
            for d in ti.static(range(self.dim)):
                c = self.cnt[None] + i * self.dim + d
                self.data_mat[0, c] = i * self.dim + d
                self.data_mat[1, c] = i * self.dim + d
                self.data_mat[2, c] = self.m[i]
                self.data_rhs[i * self.dim + d] -= self.m[i] * (self.x[i][d] - self.xTilde[i][d])
        self.cnt[None] += self.n_particles * self.dim
        # elasticity
        for e in range(self.n_elements):
            F = self.compute_T(e) @ self.restT[e].inverse()
            IB = self.restT[e].inverse()
            vol0 = self.restT[e].determinant() / self.dim / (self.dim - 1)
            dPdF = elasticity_first_piola_kirchoff_stress_derivative(F, self.la, self.mu) * dt * dt * vol0
            P = elasticity_first_piola_kirchoff_stress(F, self.la, self.mu) * dt * dt * vol0
            if ti.static(self.dim == 2):
                intermediate = ti.Matrix(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                )
                for colI in ti.static(range(4)):
                    _000 = dPdF[0, colI] * IB[0, 0]
                    _010 = dPdF[0, colI] * IB[1, 0]
                    _101 = dPdF[2, colI] * IB[0, 1]
                    _111 = dPdF[2, colI] * IB[1, 1]
                    _200 = dPdF[1, colI] * IB[0, 0]
                    _210 = dPdF[1, colI] * IB[1, 0]
                    _301 = dPdF[3, colI] * IB[0, 1]
                    _311 = dPdF[3, colI] * IB[1, 1]
                    intermediate[2, colI] = _000 + _101
                    intermediate[3, colI] = _200 + _301
                    intermediate[4, colI] = _010 + _111
                    intermediate[5, colI] = _210 + _311
                    intermediate[0, colI] = -intermediate[2, colI] - intermediate[4, colI]
                    intermediate[1, colI] = -intermediate[3, colI] - intermediate[5, colI]
                indMap = ti.Vector(
                    [
                        self.vertices[e, 0] * 2,
                        self.vertices[e, 0] * 2 + 1,
                        self.vertices[e, 1] * 2,
                        self.vertices[e, 1] * 2 + 1,
                        self.vertices[e, 2] * 2,
                        self.vertices[e, 2] * 2 + 1,
                    ]
                )
                for colI in ti.static(range(6)):
                    _000 = intermediate[colI, 0] * IB[0, 0]
                    _010 = intermediate[colI, 0] * IB[1, 0]
                    _101 = intermediate[colI, 2] * IB[0, 1]
                    _111 = intermediate[colI, 2] * IB[1, 1]
                    _200 = intermediate[colI, 1] * IB[0, 0]
                    _210 = intermediate[colI, 1] * IB[1, 0]
                    _301 = intermediate[colI, 3] * IB[0, 1]
                    _311 = intermediate[colI, 3] * IB[1, 1]
                    c = self.cnt[None] + e * 36 + colI * 6 + 0
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[2], indMap[colI], _000 + _101
                    c = self.cnt[None] + e * 36 + colI * 6 + 1
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[3], indMap[colI], _200 + _301
                    c = self.cnt[None] + e * 36 + colI * 6 + 2
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[4], indMap[colI], _010 + _111
                    c = self.cnt[None] + e * 36 + colI * 6 + 3
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[5], indMap[colI], _210 + _311
                    c = self.cnt[None] + e * 36 + colI * 6 + 4
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[0],
                        indMap[colI],
                        -_000 - _101 - _010 - _111,
                    )
                    c = self.cnt[None] + e * 36 + colI * 6 + 5
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[1],
                        indMap[colI],
                        -_200 - _301 - _210 - _311,
                    )
                self.data_rhs[self.vertices[e, 1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
                self.data_rhs[self.vertices[e, 1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
                self.data_rhs[self.vertices[e, 2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
                self.data_rhs[self.vertices[e, 2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
                self.data_rhs[self.vertices[e, 0] * 2 + 0] -= (
                    -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[1, 0] - P[0, 1] * IB[1, 1]
                )
                self.data_rhs[self.vertices[e, 0] * 2 + 1] -= (
                    -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[1, 0] - P[1, 1] * IB[1, 1]
                )
            else:
                Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                intermediate = ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z])
                for colI in ti.static(range(9)):
                    intermediate[3, colI] = (
                        IB[0, 0] * dPdF[0, colI] + IB[0, 1] * dPdF[3, colI] + IB[0, 2] * dPdF[6, colI]
                    )
                    intermediate[4, colI] = (
                        IB[0, 0] * dPdF[1, colI] + IB[0, 1] * dPdF[4, colI] + IB[0, 2] * dPdF[7, colI]
                    )
                    intermediate[5, colI] = (
                        IB[0, 0] * dPdF[2, colI] + IB[0, 1] * dPdF[5, colI] + IB[0, 2] * dPdF[8, colI]
                    )
                    intermediate[6, colI] = (
                        IB[1, 0] * dPdF[0, colI] + IB[1, 1] * dPdF[3, colI] + IB[1, 2] * dPdF[6, colI]
                    )
                    intermediate[7, colI] = (
                        IB[1, 0] * dPdF[1, colI] + IB[1, 1] * dPdF[4, colI] + IB[1, 2] * dPdF[7, colI]
                    )
                    intermediate[8, colI] = (
                        IB[1, 0] * dPdF[2, colI] + IB[1, 1] * dPdF[5, colI] + IB[1, 2] * dPdF[8, colI]
                    )
                    intermediate[9, colI] = (
                        IB[2, 0] * dPdF[0, colI] + IB[2, 1] * dPdF[3, colI] + IB[2, 2] * dPdF[6, colI]
                    )
                    intermediate[10, colI] = (
                        IB[2, 0] * dPdF[1, colI] + IB[2, 1] * dPdF[4, colI] + IB[2, 2] * dPdF[7, colI]
                    )
                    intermediate[11, colI] = (
                        IB[2, 0] * dPdF[2, colI] + IB[2, 1] * dPdF[5, colI] + IB[2, 2] * dPdF[8, colI]
                    )
                    intermediate[0, colI] = -intermediate[3, colI] - intermediate[6, colI] - intermediate[9, colI]
                    intermediate[1, colI] = -intermediate[4, colI] - intermediate[7, colI] - intermediate[10, colI]
                    intermediate[2, colI] = -intermediate[5, colI] - intermediate[8, colI] - intermediate[11, colI]
                indMap = ti.Vector(
                    [
                        self.vertices[e, 0] * 3,
                        self.vertices[e, 0] * 3 + 1,
                        self.vertices[e, 0] * 3 + 2,
                        self.vertices[e, 1] * 3,
                        self.vertices[e, 1] * 3 + 1,
                        self.vertices[e, 1] * 3 + 2,
                        self.vertices[e, 2] * 3,
                        self.vertices[e, 2] * 3 + 1,
                        self.vertices[e, 2] * 3 + 2,
                        self.vertices[e, 3] * 3,
                        self.vertices[e, 3] * 3 + 1,
                        self.vertices[e, 3] * 3 + 2,
                    ]
                )
                for rowI in ti.static(range(12)):
                    c = self.cnt[None] + e * 144 + rowI * 12 + 0
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[3],
                        IB[0, 0] * intermediate[rowI, 0]
                        + IB[0, 1] * intermediate[rowI, 3]
                        + IB[0, 2] * intermediate[rowI, 6],
                    )
                    c = self.cnt[None] + e * 144 + rowI * 12 + 1
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[4],
                        IB[0, 0] * intermediate[rowI, 1]
                        + IB[0, 1] * intermediate[rowI, 4]
                        + IB[0, 2] * intermediate[rowI, 7],
                    )
                    c = self.cnt[None] + e * 144 + rowI * 12 + 2
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[5],
                        IB[0, 0] * intermediate[rowI, 2]
                        + IB[0, 1] * intermediate[rowI, 5]
                        + IB[0, 2] * intermediate[rowI, 8],
                    )
                    c = self.cnt[None] + e * 144 + rowI * 12 + 3
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[6],
                        IB[1, 0] * intermediate[rowI, 0]
                        + IB[1, 1] * intermediate[rowI, 3]
                        + IB[1, 2] * intermediate[rowI, 6],
                    )
                    c = self.cnt[None] + e * 144 + rowI * 12 + 4
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[7],
                        IB[1, 0] * intermediate[rowI, 1]
                        + IB[1, 1] * intermediate[rowI, 4]
                        + IB[1, 2] * intermediate[rowI, 7],
                    )
                    c = self.cnt[None] + e * 144 + rowI * 12 + 5
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[8],
                        IB[1, 0] * intermediate[rowI, 2]
                        + IB[1, 1] * intermediate[rowI, 5]
                        + IB[1, 2] * intermediate[rowI, 8],
                    )
                    c = self.cnt[None] + e * 144 + rowI * 12 + 6
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[9],
                        IB[2, 0] * intermediate[rowI, 0]
                        + IB[2, 1] * intermediate[rowI, 3]
                        + IB[2, 2] * intermediate[rowI, 6],
                    )
                    c = self.cnt[None] + e * 144 + rowI * 12 + 7
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[10],
                        IB[2, 0] * intermediate[rowI, 1]
                        + IB[2, 1] * intermediate[rowI, 4]
                        + IB[2, 2] * intermediate[rowI, 7],
                    )
                    c = self.cnt[None] + e * 144 + rowI * 12 + 8
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[11],
                        IB[2, 0] * intermediate[rowI, 2]
                        + IB[2, 1] * intermediate[rowI, 5]
                        + IB[2, 2] * intermediate[rowI, 8],
                    )
                    c = self.cnt[None] + e * 144 + rowI * 12 + 9
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[0],
                        -self.data_mat[2, c - 9] - self.data_mat[2, c - 6] - self.data_mat[2, c - 3],
                    )
                    c = self.cnt[None] + e * 144 + rowI * 12 + 10
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[1],
                        -self.data_mat[2, c - 9] - self.data_mat[2, c - 6] - self.data_mat[2, c - 3],
                    )
                    c = self.cnt[None] + e * 144 + rowI * 12 + 11
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = (
                        indMap[rowI],
                        indMap[2],
                        -self.data_mat[2, c - 9] - self.data_mat[2, c - 6] - self.data_mat[2, c - 3],
                    )
                R10 = IB[0, 0] * P[0, 0] + IB[0, 1] * P[0, 1] + IB[0, 2] * P[0, 2]
                R11 = IB[0, 0] * P[1, 0] + IB[0, 1] * P[1, 1] + IB[0, 2] * P[1, 2]
                R12 = IB[0, 0] * P[2, 0] + IB[0, 1] * P[2, 1] + IB[0, 2] * P[2, 2]
                R20 = IB[1, 0] * P[0, 0] + IB[1, 1] * P[0, 1] + IB[1, 2] * P[0, 2]
                R21 = IB[1, 0] * P[1, 0] + IB[1, 1] * P[1, 1] + IB[1, 2] * P[1, 2]
                R22 = IB[1, 0] * P[2, 0] + IB[1, 1] * P[2, 1] + IB[1, 2] * P[2, 2]
                R30 = IB[2, 0] * P[0, 0] + IB[2, 1] * P[0, 1] + IB[2, 2] * P[0, 2]
                R31 = IB[2, 0] * P[1, 0] + IB[2, 1] * P[1, 1] + IB[2, 2] * P[1, 2]
                R32 = IB[2, 0] * P[2, 0] + IB[2, 1] * P[2, 1] + IB[2, 2] * P[2, 2]
                self.data_rhs[self.vertices[e, 1] * 3 + 0] -= R10
                self.data_rhs[self.vertices[e, 1] * 3 + 1] -= R11
                self.data_rhs[self.vertices[e, 1] * 3 + 2] -= R12
                self.data_rhs[self.vertices[e, 2] * 3 + 0] -= R20
                self.data_rhs[self.vertices[e, 2] * 3 + 1] -= R21
                self.data_rhs[self.vertices[e, 2] * 3 + 2] -= R22
                self.data_rhs[self.vertices[e, 3] * 3 + 0] -= R30
                self.data_rhs[self.vertices[e, 3] * 3 + 1] -= R31
                self.data_rhs[self.vertices[e, 3] * 3 + 2] -= R32
                self.data_rhs[self.vertices[e, 0] * 3 + 0] -= -R10 - R20 - R30
                self.data_rhs[self.vertices[e, 0] * 3 + 1] -= -R11 - R21 - R31
                self.data_rhs[self.vertices[e, 0] * 3 + 2] -= -R12 - R22 - R32
        self.cnt[None] += self.n_elements * (self.dim + 1) * self.dim * (self.dim + 1) * self.dim

    @ti.kernel
    def save_xPrev(self):
        for i in range(self.n_particles):
            self.xPrev[i] = self.x[i]

    @ti.kernel
    def apply_sol(self, alpha: ti.f64):
        for i in range(self.n_particles):
            for d in ti.static(range(self.dim)):
                self.x[i][d] = self.xPrev[i][d] + self.data_sol[i * self.dim + d] * alpha

    @ti.kernel
    def compute_v(self, dt: ti.f64):
        for i in range(self.n_particles):
            self.v[i] = (self.x[i] - self.xn[i]) / dt

    @ti.kernel
    def output_residual(self, dt: ti.f64) -> ti.f64:
        """
        Compute the maximum position correction from Newton solver.
        Returns |Δx|_∞ where Δx is the search direction.
        Note: convergence is checked against |Δx| < newton_v_res_tol * dt,
        which is equivalent to checking velocity-like residual |Δx|/dt < newton_v_res_tol
        """
        residual = 0.0
        for i in range(self.n_particles):
            for d in ti.static(range(self.dim)):
                residual = ti.max(residual, ti.abs(self.data_sol[i * self.dim + d]))
        return residual

    def pre_process(self):
        self.x.from_numpy(self.mesh_particles)
        self.vertices.from_numpy(self.mesh_elements)
        self.compute_restT_and_m()
        self.zero.fill(0)

        # Compute initial center of mass y position as reference for gravitational energy
        # Convert taichi fields to numpy for this computation
        m_np = self.m.to_numpy()
        x_np = self.x.to_numpy()
        total_mass = np.sum(m_np)
        com_y = np.sum(m_np * x_np[:, 1])
        self.y_ref = com_y / total_mass if total_mass > 0 else 0.0

        # Set initial velocity based on case
        if self.case == "vibration_bar":
            # Sinusoidal initial velocity: v_x = amplitude * sin(0.5 * pi * x / Lx)
            v0 = self.cfg.envs_params.get("initial_velocity_amplitude", 0.75)
            Lx = self.cfg.envs_params.Lx
            x_start = self.cfg.envs_params.get("x_start", 0.0)

            initial_v = np.zeros((self.n_particles, 2))
            for i in range(self.n_particles):
                x_rel = self.mesh_particles[i, 0] - x_start
                initial_v[i, 0] = v0 * np.sin(0.5 * np.pi * x_rel / Lx)

            self.v.from_numpy(initial_v)
        elif self.case == "twisting_column":
            # Twisting motion: create rotational velocity field around center
            # v_x = -omega * (y - y_center)
            # v_y = omega * (x - x_center)
            amplitude = self.cfg.envs_params.get("initial_twist_amplitude", 1.0)
            Lx = self.cfg.envs_params.Lx
            Ly = self.cfg.envs_params.Ly
            x_center = Lx / 2.0
            y_center = Ly / 2.0

            initial_v = np.zeros((self.n_particles, 2))
            for i in range(self.n_particles):
                x_pos = self.mesh_particles[i, 0]
                y_pos = self.mesh_particles[i, 1]
                # Rotational velocity field (counterclockwise)
                # Scale by distance from center to create differential rotation
                dx = x_pos - x_center
                dy = y_pos - y_center
                r = np.sqrt(dx * dx + dy * dy)
                # Linear velocity profile: v = omega * r, scaled by height
                y_factor = y_pos / Ly  # 0 at bottom, 1 at top
                initial_v[i, 0] = -amplitude * dy * y_factor
                initial_v[i, 1] = amplitude * dx * y_factor

            self.v.from_numpy(initial_v)
        else:
            # Default: zero initial velocity
            self.v.fill(0)

        # Set boundary conditions based on case
        if self.case == "cantilever":
            # Fix left edge (x < 1e-6)
            for i in range(self.n_particles):
                if self.mesh_particles[i, 0] < 1e-6:
                    self.fixed_nodes.append(i)
        elif self.case == "vibration_bar":
            # Fix left edge (x < x_start + small tolerance)
            x_start = self.cfg.envs_params.get("x_start", 0.0)
            Lx = self.cfg.envs_params.Lx
            bc_dist_dx = 0.05 * self.dx

            for i in range(self.n_particles):
                if self.mesh_particles[i, 0] < x_start + bc_dist_dx:
                    self.fixed_nodes.append(i)
        elif self.case == "twisting_column":
            # Fix bottom edge (y < small tolerance)
            Ly = self.cfg.envs_params.Ly
            dy = Ly / int(self.cfg.envs_params.Lx / self.dx * Ly / self.cfg.envs_params.Lx)
            bc_dist_dy = 0.05 * dy

            for i in range(self.n_particles):
                if self.mesh_particles[i, 1] < bc_dist_dy:
                    self.fixed_nodes.append(i)

        # Prepare BC DOFs
        # Initialize boundary conditions once (BCs are fixed throughout simulation)
        if not self.bc_initialized:
            bc_dof_list = []
            for node_idx in self.fixed_nodes:
                bc_dof_list.append(node_idx * self.dim)
                bc_dof_list.append(node_idx * self.dim + 1)

            # For vibration_bar: constrain all Y DOFs (1D vibration, no Y motion)
            if self.case == "vibration_bar":
                for node_idx in range(self.n_particles):
                    bc_dof_list.append(node_idx * self.dim + 1)  # Y component

            # Remove duplicates and store in Taichi field
            bc_dof_array = (
                np.unique(np.array(bc_dof_list, dtype=np.int32)) if bc_dof_list else np.array([], dtype=np.int32)
            )
            self.num_bc_dofs[None] = len(bc_dof_array)
            if len(bc_dof_array) > 0:
                self.bc_dofs.from_numpy(np.pad(bc_dof_array, (0, self.n_dof - len(bc_dof_array))))

            # Populate BC lookup array once
            self.populate_bc_lookup()
            self.bc_initialized = True

    @ti.kernel
    def compute_min_edge_length(self) -> ti.f64:
        """
        Compute minimum edge length across all elements in parallel.

        Returns:
            Minimum edge length in the mesh
        """
        min_edge = 1e10  # Large initial value

        # Parallelize over all elements
        for elem_idx in range(self.n_elements):
            # For triangular elements (3 nodes) or tetrahedral (4 nodes)
            # Compute all edge lengths
            if ti.static(self.dim == 2):
                # Triangle: 3 edges
                for i in ti.static(range(3)):
                    j = (i + 1) % 3
                    p1 = self.vertices[elem_idx, i]
                    p2 = self.vertices[elem_idx, j]
                    edge_vec = self.x[p1] - self.x[p2]
                    edge_length = edge_vec.norm()
                    ti.atomic_min(min_edge, edge_length)
            else:
                # Tetrahedron: 6 edges (0-1, 0-2, 0-3, 1-2, 1-3, 2-3)
                for i in ti.static(range(4)):
                    for j in ti.static(range(i + 1, 4)):
                        p1 = self.vertices[elem_idx, i]
                        p2 = self.vertices[elem_idx, j]
                        edge_vec = self.x[p1] - self.x[p2]
                        edge_length = edge_vec.norm()
                        ti.atomic_min(min_edge, edge_length)

        return min_edge

    @ti.kernel
    def populate_bc_lookup(self):
        """
        Populate boolean lookup array for BC DOFs (O(1) checking).
        """
        # Reset all to 0
        for i in range(self.n_dof):
            self.is_bc_dof[i] = 0

        # Mark BC DOFs as 1
        for bc_idx in range(self.num_bc_dofs[None]):
            bc_dof = self.bc_dofs[bc_idx]
            self.is_bc_dof[bc_dof] = 1

    @ti.kernel
    def fill_sparse_matrix_builder(self, builder: ti.types.sparse_matrix_builder()):
        """
        Fill the SparseMatrixBuilder directly from data_mat field with boundary conditions applied.
        Reads COO triplets from data_mat and applies Dirichlet BCs by:
        - Skipping entries in rows/cols corresponding to BC DOFs
        - Adding diagonal entries (value=1) for BC DOFs

        Uses O(1) boolean lookup for BC checking.
        """
        # First pass: Add non-BC entries from data_mat
        for idx in range(self.cnt[None]):
            row = ti.cast(self.data_mat[0, idx], ti.i32)
            col = ti.cast(self.data_mat[1, idx], ti.i32)
            val = self.data_mat[2, idx]

            # O(1) check if this row or col is a BC DOF
            if self.is_bc_dof[row] == 0 and self.is_bc_dof[col] == 0:
                builder[row, col] += val

        # Second pass: Add diagonal entries for BC DOFs
        for bc_idx in range(self.num_bc_dofs[None]):
            bc_dof = self.bc_dofs[bc_idx]
            builder[bc_dof, bc_dof] += 1.0

    @ti.kernel
    def apply_bc_to_rhs(self, rhs: ti.types.ndarray()):
        """
        Apply boundary conditions to RHS by zeroing BC DOF entries.
        (Takes a NumPy ndarray)
        """
        for bc_idx in range(self.num_bc_dofs[None]):
            bc_dof = self.bc_dofs[bc_idx]
            rhs[bc_dof] = 0.0

    def cal_dt(self):
        """
        Calculate time step using CFL condition for elastic wave propagation.

        For elastic materials, the wave speed (sound speed) is:
            c = sqrt((lambda + 2*mu) / rho)

        CFL condition:
            dt = CFL * dx / c

        where dx is the characteristic element size.
        """
        # Calculate wave speed for elastic material
        # c = sqrt((lambda + 2*mu) / rho)
        wave_speed = np.sqrt((self.la + 2 * self.mu) / self.density)

        # Compute minimum edge length using Taichi kernel (parallel)
        min_edge_length = self.compute_min_edge_length()

        # Calculate dt from CFL condition
        dt_cfl = self.cfl * min_edge_length / wave_speed

        if self.verbose:
            print(f"CFL-based time step calculation:")
            print(f"  Wave speed: {wave_speed:.2e} m/s")
            print(f"  Min element size: {min_edge_length:.2e} m")
            print(f"  CFL number: {self.cfl}")
            print(f"  Computed dt: {dt_cfl:.2e} s")

        return dt_cfl

    def step(self, dt):
        # Initialize state for this timestep
        self.compute_xn_and_xTilde(dt)

        # Newton iteration loop
        for newton_iter in range(self.max_newton_iter):
            # Reset system matrices and vectors
            self.data_mat.fill(0)
            self.data_rhs.fill(0)
            self.data_sol.fill(0)

            # Assemble Hessian and gradient
            self.compute_hessian_and_gradient(dt)
            self.cnt_hessian_gradient += 1

            if self.verbose:
                print(f"  Newton iter {newton_iter}: Total entries: {self.cnt[None]}")

            # Build sparse matrix directly from data_mat field with BCs applied
            # Maximum triplets = original entries + diagonal BC entries
            max_triplets = self.cnt[None] + self.num_bc_dofs[None]
            builder = ti.linalg.SparseMatrixBuilder(self.n_dof, self.n_dof, max_num_triplets=max_triplets)
            self.fill_sparse_matrix_builder(builder)
            A = builder.build()
            b = self.data_rhs.to_numpy()
            if self.num_bc_dofs[None] > 0:
                self.apply_bc_to_rhs(b)
            if not self.sparsity_pattern_analyzed:
                self.solver.analyze_pattern(A)
                self.sparsity_pattern_analyzed = True
                if self.verbose:
                    print("Sparsity pattern analyzed ONCE.")

            self.solver.factorize(A)  # This STAYS inside the loop

            x = self.solver.solve(b)
            self.data_sol.from_numpy(x)

            # Even simpler, the old code's from_numpy(x) is fine:
            self.data_sol.from_numpy(x)
            self.cnt_linear_solve += 1

            # DEBUG: Check if copy worked correctly
            if self.verbose and newton_iter == 0:
                data_sol_np = self.data_sol.to_numpy()
                print(
                    f"  DEBUG: After copy - min: {data_sol_np.min():.6e}, max: {data_sol_np.max():.6e}, has_nan: {np.isnan(data_sol_np).any()}"
                )
                print(f"  DEBUG: First 5 data_sol values: {data_sol_np[:5]}")

            # Check convergence criterion
            # residual is |Δx|_∞, we check if |Δx|/dt < newton_v_res_tol
            residual = self.output_residual(dt)
            if self.verbose:
                print(f"  Newton iter {newton_iter}: |Δx| = {residual:.6e}, |Δx|/dt = {residual/dt:.6e}")

            converged = residual < self.newton_v_res_tol * dt

            # Line search: ensure energy decreases (always perform, even if converged)
            E0 = self.compute_energy(dt)
            self.cnt_energy += 1
            self.save_xPrev()
            alpha = 1.0
            self.apply_sol(alpha)
            E = self.compute_energy(dt)
            self.cnt_energy += 1

            line_search_iter = 0
            while E > E0 and line_search_iter < 20:
                alpha *= 0.5
                self.apply_sol(alpha)
                E = self.compute_energy(dt)
                self.cnt_energy += 1
                line_search_iter += 1

            if self.verbose and line_search_iter > 0:
                print(f"  Line search: {line_search_iter} iters, alpha = {alpha}")

            # After applying solution, check if we should stop iterating
            if converged:
                if self.verbose:
                    print(f"  Newton converged at iteration {newton_iter}")
                break

        # After Newton loop (converged or max iterations reached), update velocity
        self.compute_v(dt)

    def dump(self):
        # Log energies for convergence analysis
        self.log_energies()

        particle_pos = self.x.to_numpy()
        particle_vel = self.v.to_numpy()

        if self.dim == 2:
            particle_pos = np.hstack([particle_pos, np.zeros((self.n_particles, 1))])
            particle_vel = np.hstack([particle_vel, np.zeros((self.n_particles, 1))])

        # Create boundary condition flag (1 = Dirichlet BC, 0 = free)
        is_bc = np.zeros(self.n_particles)
        for node_idx in self.fixed_nodes:
            is_bc[node_idx] = 1.0

        point_data = {"velocity": particle_vel, "is_bc": is_bc}

        mesh = meshio.Mesh(points=particle_pos, cells={"triangle": self.mesh_elements}, point_data=point_data)
        meshio.write(f"{self.dump_dir}/frame_{self.record_frame:06d}.vtk", mesh)

    def post_process(self):
        """
        Calculate total computational cost based on operation counts.

        Cost breakdown (per element operation counts for 2D):
        - Hessian+Gradient assembly: ~804 ops per element per call
          (includes SVD, stress derivative with 4D tensor, assembly)
        - Energy evaluation: ~54 ops per element per call
          (includes SVD, energy function)
        - Sparse linear solve: O((n_particles * dim)^2) per call

        Detailed per-element costs:
        - compute_energy: compute_T(4) + matmul(12) + vol0(3) + SVD(25) + energy(10) = 54
        - compute_hessian_gradient: compute_T(4) + matmul(12) + vol0(3) + dPdF(330)
          + P(35) + intermediate_assembly(72) + hessian_assembly(312) + gradient_assembly(36) = 804

        Total cost = cnt_hessian_gradient * n_elements * 804
                   + cnt_energy * n_elements * 54
                   + cnt_linear_solve * (n_particles * dim)^2
        """
        n_dof = self.n_particles * self.dim

        # Cost constants per element (for 2D)
        cost_per_hessian_gradient = 804  # operations per element
        cost_per_energy = 54  # operations per element

        # normalize by cost_per_energy (seeing it as unit cost) to keep numbers manageable
        cost_hessian_gradient = (
            self.cnt_hessian_gradient * self.n_elements * cost_per_hessian_gradient / cost_per_energy
        )
        cost_energy = self.cnt_energy * self.n_elements * cost_per_energy / cost_per_energy
        cost_linear_solve = self.cnt_linear_solve * n_dof * n_dof / cost_per_energy

        total_cost = cost_hessian_gradient + cost_energy + cost_linear_solve

        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            meta = {
                "cost": int(total_cost),
                "cost_breakdown": {
                    "hessian_gradient": int(cost_hessian_gradient),
                    "energy": int(cost_energy),
                    "linear_solve": int(cost_linear_solve),
                },
                "operation_counts": {
                    "cnt_hessian_gradient": int(self.cnt_hessian_gradient),
                    "cnt_energy": int(self.cnt_energy),
                    "cnt_linear_solve": int(self.cnt_linear_solve),
                },
                "total_steps": int(self.num_steps),
                "n_particles": int(self.n_particles),
                "n_elements": int(self.n_elements),
                "n_dof": int(n_dof),
            }
            import json

            json.dump(meta, f, indent=4)

        # Save energies for convergence analysis
        energies_path = os.path.join(self.dump_dir, "energies.npz")
        np.savez(
            energies_path,
            kin=np.array(self.energy_log_kin),
            pot=np.array(self.energy_log_pot),
            gra=np.array(self.energy_log_gra),
            tot=np.array(self.energy_log_tot),
        )

        # Plot energies vs time
        paint_res = 300
        plt.clf()
        plt.figure(figsize=(8, 4))
        plt.rcParams["figure.dpi"] = paint_res
        plt.rcParams["font.family"] = "Times New Roman"
        plt.xlabel(r"$t[s]$")
        plt.ylabel(r"$E[J]$")

        # Set legend
        legend = [r"$E_{kin}$", r"$E_{pot}$", r"$E_{grav}$", r"$E_{tot}$"]

        # Time series
        time = np.arange(len(self.energy_log_kin)) * self.cfg.record_dt

        # Plot all energy components
        plt.plot(time, self.energy_log_kin, label=legend[0])
        plt.plot(time, self.energy_log_pot, label=legend[1])
        plt.plot(time, self.energy_log_gra, label=legend[2])
        plt.plot(time, self.energy_log_tot, label=legend[3])

        plt.legend(loc="best")
        plt.savefig(os.path.join(self.dump_dir, "energies.png"), dpi=paint_res)
        plt.close()

        if self.verbose:
            print(f"\nCost Summary:")
            print(
                f"  Hessian+Gradient assemblies: {self.cnt_hessian_gradient} × {self.n_elements} × {cost_per_hessian_gradient} = {cost_hessian_gradient}"
            )
            print(f"  Energy evaluations: {self.cnt_energy} × {self.n_elements} × {cost_per_energy} = {cost_energy}")
            print(f"  Linear solves: {self.cnt_linear_solve} × {n_dof}^2 = {cost_linear_solve}")
            print(f"  Total cost: {total_cost}")
            print(f"  Average Newton iters per step: {self.cnt_hessian_gradient / self.num_steps:.2f}")
            print(f"\nEnergy conservation:")
            if len(self.energy_log_tot) > 1:
                energy_variation = np.std(self.energy_log_tot) / (np.mean(np.abs(self.energy_log_tot)) + 1e-12)
                print(f"  Total energy variation (CV): {energy_variation:.2e}")
                print(f"  Energy range: {np.min(self.energy_log_tot):.6e} to {np.max(self.energy_log_tot):.6e}")
