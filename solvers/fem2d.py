import taichi as ti
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import meshio
import os

from .base_solver import SIMULATOR
from .fastipc_utils.common.physics.fixed_corotated import *
from .fastipc_utils.common.math.math_tools import *


@ti.data_oriented
class FEM2D(SIMULATOR):
    def __init__(self, verbose, cfg):
        super().__init__(verbose, cfg)

        self.dim = cfg.dim
        self.E = cfg.E
        self.nu = cfg.nu
        self.la = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
        self.density = cfg.density
        self.dt = cfg.dt
        self.nx = cfg.nx

        # Case type and gravity
        self.case = cfg.get('case', 'cantilever')
        self.gravity = cfg.envs_params.get('gravity', -9.8)

        # Newton solver parameters
        self.max_newton_iter = cfg.max_newton_iter
        self.newton_residual_threshold = cfg.newton_residual_threshold

        mesh = self._get_mesh(self.nx, cfg)
        self.mesh_particles = mesh.points
        self.mesh_elements = mesh.cells[0].data
        self.mesh_scale = cfg.mesh_scale
        self.mesh_offset = cfg.mesh_offset

        self.n_particles = len(self.mesh_particles)
        self.n_elements = len(self.mesh_elements)

        self.real = ti.f64
        ti.init(arch=ti.cpu, default_fp=self.real, verbose=self.verbose)

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

        self.data_rhs = ti.field(self.real, shape=self.n_particles * self.dim)
        self.data_mat = ti.field(self.real, shape=(3, 2000000))
        self.data_sol = ti.field(self.real, shape=self.n_particles * self.dim)
        
        self.dump_dir = cfg.dump_dir
        self.fixed_nodes = []
        if not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir)

    def _get_mesh(self, nx, cfg):
        mesh_path = f"output/mesh/{self.case}_nx{nx}.obj"
        if os.path.exists(mesh_path):
            return meshio.read(mesh_path)

        Lx = cfg.envs_params.Lx
        Ly = cfg.envs_params.Ly
        n_gx_e = nx
        n_gy_e = int(nx * Ly / Lx)
        dx = Lx / n_gx_e
        dy = Ly / n_gy_e

        # Get starting position based on case
        if self.case == 'vibration_bar':
            x_start = cfg.envs_params.get('x_start', 0.0)
            y_start = cfg.envs_params.get('y_start', 0.0)
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
                total_energy += elasticity_energy(ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]]), self.la, self.mu) * dt * dt * vol0
        return total_energy

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
                intermediate = ti.Matrix([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
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
                indMap = ti.Vector([self.vertices[e, 0] * 2, self.vertices[e, 0] * 2 + 1,
                                    self.vertices[e, 1] * 2, self.vertices[e, 1] * 2 + 1,
                                    self.vertices[e, 2] * 2, self.vertices[e, 2] * 2 + 1])
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
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[0], indMap[colI], - _000 - _101 - _010 - _111
                    c = self.cnt[None] + e * 36 + colI * 6 + 5
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[1], indMap[colI], - _200 - _301 - _210 - _311
                self.data_rhs[self.vertices[e, 1] * 2 + 0] -= P[0, 0] * IB[0, 0] + P[0, 1] * IB[0, 1]
                self.data_rhs[self.vertices[e, 1] * 2 + 1] -= P[1, 0] * IB[0, 0] + P[1, 1] * IB[0, 1]
                self.data_rhs[self.vertices[e, 2] * 2 + 0] -= P[0, 0] * IB[1, 0] + P[0, 1] * IB[1, 1]
                self.data_rhs[self.vertices[e, 2] * 2 + 1] -= P[1, 0] * IB[1, 0] + P[1, 1] * IB[1, 1]
                self.data_rhs[self.vertices[e, 0] * 2 + 0] -= -P[0, 0] * IB[0, 0] - P[0, 1] * IB[0, 1] - P[0, 0] * IB[1, 0] - P[0, 1] * IB[1, 1]
                self.data_rhs[self.vertices[e, 0] * 2 + 1] -= -P[1, 0] * IB[0, 0] - P[1, 1] * IB[0, 1] - P[1, 0] * IB[1, 0] - P[1, 1] * IB[1, 1]
            else:
                Z = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                intermediate = ti.Matrix.rows([Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z])
                for colI in ti.static(range(9)):
                    intermediate[3, colI] = IB[0, 0] * dPdF[0, colI] + IB[0, 1] * dPdF[3, colI] + IB[0, 2] * dPdF[6, colI]
                    intermediate[4, colI] = IB[0, 0] * dPdF[1, colI] + IB[0, 1] * dPdF[4, colI] + IB[0, 2] * dPdF[7, colI]
                    intermediate[5, colI] = IB[0, 0] * dPdF[2, colI] + IB[0, 1] * dPdF[5, colI] + IB[0, 2] * dPdF[8, colI]
                    intermediate[6, colI] = IB[1, 0] * dPdF[0, colI] + IB[1, 1] * dPdF[3, colI] + IB[1, 2] * dPdF[6, colI]
                    intermediate[7, colI] = IB[1, 0] * dPdF[1, colI] + IB[1, 1] * dPdF[4, colI] + IB[1, 2] * dPdF[7, colI]
                    intermediate[8, colI] = IB[1, 0] * dPdF[2, colI] + IB[1, 1] * dPdF[5, colI] + IB[1, 2] * dPdF[8, colI]
                    intermediate[9, colI] = IB[2, 0] * dPdF[0, colI] + IB[2, 1] * dPdF[3, colI] + IB[2, 2] * dPdF[6, colI]
                    intermediate[10, colI] = IB[2, 0] * dPdF[1, colI] + IB[2, 1] * dPdF[4, colI] + IB[2, 2] * dPdF[7, colI]
                    intermediate[11, colI] = IB[2, 0] * dPdF[2, colI] + IB[2, 1] * dPdF[5, colI] + IB[2, 2] * dPdF[8, colI]
                    intermediate[0, colI] = -intermediate[3, colI] - intermediate[6, colI] - intermediate[9, colI]
                    intermediate[1, colI] = -intermediate[4, colI] - intermediate[7, colI] - intermediate[10, colI]
                    intermediate[2, colI] = -intermediate[5, colI] - intermediate[8, colI] - intermediate[11, colI]
                indMap = ti.Vector([self.vertices[e, 0] * 3, self.vertices[e, 0] * 3 + 1, self.vertices[e, 0] * 3 + 2,
                                    self.vertices[e, 1] * 3, self.vertices[e, 1] * 3 + 1, self.vertices[e, 1] * 3 + 2,
                                    self.vertices[e, 2] * 3, self.vertices[e, 2] * 3 + 1, self.vertices[e, 2] * 3 + 2,
                                    self.vertices[e, 3] * 3, self.vertices[e, 3] * 3 + 1, self.vertices[e, 3] * 3 + 2])
                for rowI in ti.static(range(12)):
                    c = self.cnt[None] + e * 144 + rowI * 12 + 0
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[3], IB[0, 0] * intermediate[rowI, 0] + IB[0, 1] * intermediate[rowI, 3] + IB[0, 2] * intermediate[rowI, 6]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 1
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[4], IB[0, 0] * intermediate[rowI, 1] + IB[0, 1] * intermediate[rowI, 4] + IB[0, 2] * intermediate[rowI, 7]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 2
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[5], IB[0, 0] * intermediate[rowI, 2] + IB[0, 1] * intermediate[rowI, 5] + IB[0, 2] * intermediate[rowI, 8]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 3
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[6], IB[1, 0] * intermediate[rowI, 0] + IB[1, 1] * intermediate[rowI, 3] + IB[1, 2] * intermediate[rowI, 6]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 4
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[7], IB[1, 0] * intermediate[rowI, 1] + IB[1, 1] * intermediate[rowI, 4] + IB[1, 2] * intermediate[rowI, 7]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 5
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[8], IB[1, 0] * intermediate[rowI, 2] + IB[1, 1] * intermediate[rowI, 5] + IB[1, 2] * intermediate[rowI, 8]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 6
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[9], IB[2, 0] * intermediate[rowI, 0] + IB[2, 1] * intermediate[rowI, 3] + IB[2, 2] * intermediate[rowI, 6]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 7
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[10], IB[2, 0] * intermediate[rowI, 1] + IB[2, 1] * intermediate[rowI, 4] + IB[2, 2] * intermediate[rowI, 7]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 8
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[11], IB[2, 0] * intermediate[rowI, 2] + IB[2, 1] * intermediate[rowI, 5] + IB[2, 2] * intermediate[rowI, 8]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 9
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[0], -self.data_mat[2, c - 9] - self.data_mat[2, c - 6] - self.data_mat[2, c - 3]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 10
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[1], -self.data_mat[2, c - 9] - self.data_mat[2, c - 6] - self.data_mat[2, c - 3]
                    c = self.cnt[None] + e * 144 + rowI * 12 + 11
                    self.data_mat[0, c], self.data_mat[1, c], self.data_mat[2, c] = indMap[rowI], indMap[2], -self.data_mat[2, c - 9] - self.data_mat[2, c - 6] - self.data_mat[2, c - 3]
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
    def apply_sol(self, alpha : ti.f64):
        for i in range(self.n_particles):
            for d in ti.static(range(self.dim)):
                self.x[i][d] = self.xPrev[i][d] + self.data_sol[i * self.dim + d] * alpha

    @ti.kernel
    def compute_v(self, dt: ti.f64):
        for i in range(self.n_particles):
            self.v[i] = (self.x[i] - self.xn[i]) / dt

    @ti.kernel
    def output_residual(self, dt: ti.f64) -> ti.f64:
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

        # Set initial velocity based on case
        if self.case == 'vibration_bar':
            # Sinusoidal initial velocity: v_x = amplitude * sin(0.5 * pi * x / Lx)
            v0 = self.cfg.envs_params.get('initial_velocity_amplitude', 0.75)
            Lx = self.cfg.envs_params.Lx
            x_start = self.cfg.envs_params.get('x_start', 0.0)

            initial_v = np.zeros((self.n_particles, 2))
            for i in range(self.n_particles):
                x_rel = self.mesh_particles[i, 0] - x_start
                initial_v[i, 0] = v0 * np.sin(0.5 * np.pi * x_rel / Lx)

            self.v.from_numpy(initial_v)
        else:
            # Default: zero initial velocity
            self.v.fill(0)

        # Set boundary conditions based on case
        if self.case == 'cantilever':
            # Fix left edge (x < 1e-6)
            for i in range(self.n_particles):
                if self.mesh_particles[i, 0] < 1e-6:
                    self.fixed_nodes.append(i)
        elif self.case == 'vibration_bar':
            # Fix left edge (x < x_start + small tolerance)
            x_start = self.cfg.envs_params.get('x_start', 0.0)
            Lx = self.cfg.envs_params.Lx
            dx = Lx / self.nx
            bc_dist_dx = 0.05 * dx

            for i in range(self.n_particles):
                if self.mesh_particles[i, 0] < x_start + bc_dist_dx:
                    self.fixed_nodes.append(i)

    def cal_dt(self):
        return self.dt

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

            if self.verbose:
                print(f"  Newton iter {newton_iter}: Total entries: {self.cnt[None]}")

            # Convert to sparse matrix format
            mat = self.data_mat.to_numpy()
            row, col, val = mat[0, :self.cnt[None]], mat[1, :self.cnt[None]], mat[2, :self.cnt[None]]
            rhs = self.data_rhs.to_numpy()
            n = self.n_particles * self.dim
            A = scipy.sparse.csr_matrix((val, (row, col)), shape=(n, n))
            A = scipy.sparse.lil_matrix(A)

            # Apply boundary conditions (Dirichlet BC on fixed nodes)
            D = []
            for node_idx in self.fixed_nodes:
                D.append(node_idx * self.dim)
                D.append(node_idx * self.dim + 1)
            D = np.array(D, dtype=np.int32)

            # Only apply BC if there are fixed nodes
            if len(D) > 0:
                A[:, D] = 0
                A[D, :] = 0
                A = scipy.sparse.csr_matrix(A)
                A += scipy.sparse.csr_matrix((np.ones(len(D)), (D, D)), shape=(n, n))
                rhs[D] = 0
            else:
                A = scipy.sparse.csr_matrix(A)

            # Solve linear system
            self.data_sol.from_numpy(scipy.sparse.linalg.spsolve(A, rhs))

            # Check convergence criterion
            residual = self.output_residual(dt)
            if self.verbose:
                print(f'  Newton iter {newton_iter}: residual = {residual}')

            converged = residual < self.newton_residual_threshold * dt

            # Line search: ensure energy decreases (always perform, even if converged)
            E0 = self.compute_energy(dt)
            self.save_xPrev()
            alpha = 1.0
            self.apply_sol(alpha)
            E = self.compute_energy(dt)

            line_search_iter = 0
            while E > E0 and line_search_iter < 20:
                alpha *= 0.5
                self.apply_sol(alpha)
                E = self.compute_energy(dt)
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
        particle_pos = self.x.to_numpy()
        particle_vel = self.v.to_numpy()

        if self.dim == 2:
            particle_pos = np.hstack([particle_pos, np.zeros((self.n_particles, 1))])
            particle_vel = np.hstack([particle_vel, np.zeros((self.n_particles, 1))])

        # Create boundary condition flag (1 = Dirichlet BC, 0 = free)
        is_bc = np.zeros(self.n_particles)
        for node_idx in self.fixed_nodes:
            is_bc[node_idx] = 1.0

        point_data = {
            'velocity': particle_vel,
            'is_bc': is_bc
        }

        mesh = meshio.Mesh(points=particle_pos, cells={'triangle': self.mesh_elements}, point_data=point_data)
        meshio.write(f'{self.dump_dir}/frame_{self.record_frame:06d}.vtk', mesh)

    def post_process(self):
        cost = self.num_steps * self.n_particles * self.n_elements
        with open(os.path.join(self.dump_dir, "meta.json"), "w") as f:
            meta = {
                "cost": cost,
                "total_steps": int(self.num_steps),
            }
            import json
            json.dump(meta, f, indent=4)
        if self.verbose:
            print(f"Run cost: {cost}")