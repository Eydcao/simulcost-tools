import taichi as ti
import numpy as np
import pickle
import os
import meshio
from scipy.spatial import Delaunay
import scipy.interpolate as interpolate
from scipy.ndimage import gaussian_filter1d
from helpers import barycentric_coord_2d, barycentric_coord_3d, mesh2line, build_connectivity, B_spline, ADV_TYPE, ADV_TYPE_STR
from matplotlib import cm
import matplotlib.pyplot as plt

verbose = False
output_png = False
output_vtk = False

advect_scheme = ADV_TYPE.FLIP
quality = 2.0  # 0.5,1,2,5

DIM = 2
end_time = 4
dt = 0.001 / quality
end_frame = int(end_time / dt)
frame_dt = dt
eps = 1e-5  # for seperation factor
gravity = -9.81
Lx, Ly = 11.0, 8.0
# _e means effective, only in these cells there are particles
n_gx, n_gx_e, p_perx = int(22 * quality), int(20 * quality), 2
n_gy, n_gy_e, p_pery = int(16 * quality), int(4 * quality), 2
n_particles = int(n_gx_e * n_gy_e * p_perx * p_pery)
dx, dy = Lx / n_gx, Ly / n_gy
p_vol, p_rho = dx*dy/p_perx/p_pery, 2.0
p_mass = p_vol * p_rho
E, nu = 1.0e5, 0.29  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

bc_dist_dx, bc_dist_dy = 0.05 * dx, 0.05 * dy

hash_dx, hash_dy = dx * 2.0, dy * 2.0

# ti.init(arch=ti.cpu, debug=True, default_fp=ti.float64, cpu_max_num_threads=1, random_seed=1000)  # CPU debug
ti.init(arch=ti.cpu, debug=False, default_fp=ti.float64, random_seed=1000)  # CPU
# ti.init(arch=ti.gpu, debug=False, default_fp=ti.float32, random_seed=1000)  # GPU; float 64 on some GPU causes bug


def init_particles():
    # vibrating bar
    # create containers for particle attributes
    x = np.zeros((n_particles, DIM), dtype=float)
    v = np.zeros((n_particles, DIM), dtype=float)
    material = np.zeros(n_particles, dtype=int)
    F = np.zeros((n_particles, DIM, DIM), dtype=float)
    Jp = np.zeros(n_particles, dtype=float)
    dxp = dx / p_perx
    dyp = dy / p_pery
    x_start = 0.0
    y_start = 5.0
    for i in range(n_gx_e * p_perx):
        for j in range(n_gy_e * p_pery):
            idx = i * n_gy_e * p_pery + j
            x[idx] = np.array([x_start + dxp * 0.5 + dxp * i, y_start + dyp * 0.5 + dyp * j])
            material[idx] = 1  # 0: fluid 1: jelly 2: snow
            v[idx] = np.array([0, 0])
            F[idx] = np.array([[1, 0], [0, 1]])
            Jp[idx] = 1

    return x, material, v, F, Jp


def init_grid_geometry():
    # grid is a 2d array of vertices
    # shape is Lx * Ly
    # grid number is n_gy, p_pery
    # grid resolution is dx, dy
    v_pos = np.zeros(((n_gx + 1) * (n_gy + 1), DIM), dtype=float)
    for i in range(n_gx + 1):
        for j in range(n_gy + 1):
            v_pos[i * (n_gy + 1) + j] = np.array([dx * i, dy * j])

    # cell
    D = Delaunay(v_pos)
    cell = D.simplices

    return v_pos, cell


p_pos, p_mat, p_vel, p_F, p_Jp = init_particles()
v_pos, cell, = init_grid_geometry()
f_1b_v_idx, f_1b_v_var, max_nonzero_f_1b_v, f_1b_v_connectivity_2_f_v, hash2cell_idx, hash2cell_var, hash_min, hash_max = build_connectivity(v_pos, cell, np.array([hash_dx, hash_dy]))

num_v = v_pos.shape[0]
num_cell = cell.shape[0]

hash_stride = np.cumprod((hash_max - hash_min)[::-1])[::-1]
hash_stride[:-1] = hash_stride[1:]
hash_stride[-1] = 1

# post script
# find the indices of initial particle pos that are closest to [0.476, 0.26]
sample_pos = np.array([10.0, 5.0 + 1.0])
sample_idx = np.argmin(np.linalg.norm(p_pos - sample_pos, axis=1))
sampled_pos = p_pos[sample_idx]
# print('sampled_x is ', sampled_pos)
# exit()

# particle attributes
x = ti.Vector.field(DIM, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(DIM, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(DIM, DIM, dtype=float, shape=n_particles)  # affine velocity field
B = ti.Matrix.field(DIM, DIM, dtype=float, shape=n_particles)
F = ti.Matrix.field(DIM, DIM, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
Dp_inv = ti.Matrix.field(DIM, DIM, dtype=float, shape=n_particles)
# p_in_g information
p_in_cell_idx = ti.field(dtype=int, shape=n_particles)
p_in_cell_bcoord = ti.Vector.field(DIM + 1, dtype=float, shape=n_particles)  # barycentric coord
# p_v_ws's shape is (num_v, max_nonzero_f_1b_v), as a particle can only appear in one cell and interact with all of its 1b adj vertices
p_v_ws = ti.field(dtype=float, shape=(n_particles, max_nonzero_f_1b_v))

# static connectivity
ti_cell = ti.Vector.field(DIM + 1, dtype=int, shape=num_cell)
ti_f_1b_v_idx = ti.field(dtype=int, shape=f_1b_v_idx.shape[0])
ti_f_1b_v_var = ti.field(dtype=int, shape=f_1b_v_var.shape[0])
ti_f_1b_v_connectivity_2_f_v = ti.Vector.field(DIM + 1, dtype=float, shape=f_1b_v_connectivity_2_f_v.shape[0])

# v attributes: pos, v3l, m, vel, dirichelet flag (TODO set dirichelet flag)
# static
ti_v_pos = ti.Vector.field(DIM, dtype=float, shape=num_v)
ti_v_dirichlet = ti.field(dtype=int, shape=num_v)
# dynamic
ti_v_m = ti.field(dtype=float, shape=num_v)
ti_v_vel = ti.Vector.field(DIM, dtype=float, shape=num_v)
ti_v_vel_old = ti.Vector.field(DIM, dtype=float, shape=num_v)  # can be either v_new or delta_v; only used in flip

# hash map attributes
ti_hash2cell_idx = ti.field(dtype=int, shape=hash2cell_idx.shape[0])
ti_hash2cell_var = ti.field(dtype=int, shape=hash2cell_var.shape[0])
ti_hash_min = ti.Vector.field(DIM, dtype=int, shape=1)
ti_hash_max = ti.Vector.field(DIM, dtype=int, shape=1)
ti_hash_stride = ti.Vector.field(DIM, dtype=int, shape=1)

# summed vars
L_A_p = ti.Vector.field(DIM + 1, dtype=float, shape=1)
L_A_g = ti.Vector.field(DIM + 1, dtype=float, shape=1)

# post script fields
pot_eng = ti.field(dtype=float, shape=1)
kin_eng = ti.field(dtype=float, shape=1)
gra_eng = ti.field(dtype=float, shape=1)


def init_mesh():
    x.from_numpy(p_pos)
    print('init particle pos done')
    material.from_numpy(p_mat)
    v.from_numpy(p_vel)
    F.from_numpy(p_F)
    Jp.from_numpy(p_Jp)
    print('init particle attributes done')
    ti_v_pos.from_numpy(v_pos)
    print('init v pos done')
    ti_cell.from_numpy(cell)
    ti_f_1b_v_idx.from_numpy(f_1b_v_idx)
    ti_f_1b_v_var.from_numpy(f_1b_v_var)
    ti_f_1b_v_connectivity_2_f_v.from_numpy(f_1b_v_connectivity_2_f_v.astype(float))
    print('init cell connectivity done')
    # set_dirichelet TODO
    print('set dirichelet done')
    ti_hash2cell_idx.from_numpy(hash2cell_idx)
    ti_hash2cell_var.from_numpy(hash2cell_var)
    ti_hash_min.from_numpy(np.array([hash_min]))
    ti_hash_max.from_numpy(np.array([hash_max]))
    ti_hash_stride.from_numpy(np.array([hash_stride]))
    print('set hash map done')


@ti.func
def hashing(spatial_idx: ti.template()):
    res = (spatial_idx - ti_hash_min[0]) * ti_hash_stride[0]
    ws = ti.Vector.one(int, DIM)

    return res.dot(ws)


@ti.kernel
def zero_clear():
    # zero all elements in v containers
    for v in ti_v_m:
        ti_v_m[v] *= 0
        ti_v_vel_old[v] *= 0
        ti_v_vel[v] *= 0
    # zero all p_v_ws
    for p, b in p_v_ws:
        p_v_ws[p, b] = 0

    # post
    pot_eng[0] = 0.0
    kin_eng[0] = 0.0
    gra_eng[0] = 0.0
    # post


@ti.kernel
def call_back(t: float):
    pass


@ti.kernel
def cal_DP_inv():
    # Dp_inv calculation
    # NOTE BTW, calculate p_in_mesh informations, re-use them in later g2p or p2g processes
    for p in x:
        # force search
        found_cell = False
        # find which hash box the particle is in
        hash_vec = ti.Vector([hash_dx, hash_dy])
        hash_spatial_idx = ti.floor(x[p] / hash_vec, dtype=int)
        hash_idx = hashing(hash_spatial_idx)
        # only iterate through the cells inters with this hash box
        inter_cell_start = ti_hash2cell_idx[hash_idx]
        inter_cell_end = ti_hash2cell_idx[hash_idx + 1]
        for tidx in range(inter_cell_start, inter_cell_end):
            f = ti_hash2cell_var[tidx]
            v0 = ti_v_pos[ti_cell[f][0]]
            v1 = ti_v_pos[ti_cell[f][1]]
            v2 = ti_v_pos[ti_cell[f][2]]
            # v3 = ti_v_pos[ti_cell[f][3]]
            P = x[p]
            # get the max and min of the v0 to v3
            # use bbox to filter
            min_bb = ti.min(v0, v1, v2)
            max_bb = ti.max(v0, v1, v2)
            if all(P >= min_bb - eps) and all(P <= max_bb + eps):
                # get the b_coord of this p in this cell
                # if all larger than zero, then this p is in this cell
                b_coord = barycentric_coord_2d(v0, v1, v2, P)
                if all(b_coord >= -eps):
                    p_in_cell_idx[p] = f
                    p_in_cell_bcoord[p] = b_coord
                    found_cell = True
                    break
        if not found_cell:
            print('error not found cell')
            print('p, p_pos ', p, x[p])
            p_in_cell_idx[p] = -1
        else:
            # get which cell this p is in
            cell_idx = p_in_cell_idx[p]
            b_coord = p_in_cell_bcoord[p]
            # calculate the p_v_ws using moving least square on the fly
            x_p = x[p]
            adj_v_idx_start = ti_f_1b_v_idx[cell_idx]
            adj_v_idx_end = ti_f_1b_v_idx[cell_idx + 1]
            # edge case 1: smaller than dim+1 adj vertices, try uniform weights
            # NOTE but this case should never (precondition) happen because it degenerates
            if adj_v_idx_end - adj_v_idx_start < DIM + 1:
                for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                    p_v_ws[p, adj_i - adj_v_idx_start] = 1.0 / (adj_v_idx_end - adj_v_idx_start)
            # edge case 2: if equal to DIM + 1, use barrycentric coord
            # NOTE this case only happens when there is a single element
            elif adj_v_idx_end - adj_v_idx_start == DIM + 1:
                for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                    p_v_ws[p, adj_i - adj_v_idx_start] = b_coord[adj_i - adj_v_idx_start]
            else:
                # calculate the M = ETDInv @ E by summing up the dpos/D outproduct with dpos
                # since taichi doesnot support dynamic matrix
                M = ti.Matrix.zero(float, DIM + 1, DIM + 1)
                for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                    v_idx = ti_f_1b_v_var[adj_i]
                    v_connect_2_cell_v = ti_f_1b_v_connectivity_2_f_v[adj_i]
                    dpos = ti_v_pos[v_idx] - x_p
                    d_hat = dpos.norm() / dx
                    invD = B_spline(d_hat)
                    # get the sum of product of v_connect_2_cell_v and b_coord
                    temp = v_connect_2_cell_v * b_coord
                    s = temp.sum()
                    # diminish invD by s
                    invD *= s
                    invD = ti.max(invD, eps)
                    E = ti.Vector.one(float, DIM + 1)
                    # assign dpos to trailing dim of E
                    for d in ti.static(range(DIM)):
                        E[d + 1] = dpos[d]
                    M += (E * invD).outer_product(E)
                MInv = M.inverse()
                # calculate the p_v_ws = MInv @ ETDInv's col(i) by iterating through all adj vertices
                # since taichi doesnot support dynamic matrix
                # w_sum = 0.0
                for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                    v_idx = ti_f_1b_v_var[adj_i]
                    v_connect_2_cell_v = ti_f_1b_v_connectivity_2_f_v[adj_i]
                    dpos = ti_v_pos[v_idx] - x_p
                    # sepeartion factor D
                    d_hat = dpos.norm() / dx
                    invD = B_spline(d_hat)
                    # get the sum of product of v_connect_2_cell_v and b_coord
                    temp = v_connect_2_cell_v * b_coord
                    s = temp.sum()
                    # diminish invD by s
                    invD *= s
                    invD = ti.max(invD, eps)
                    E = ti.Vector.one(float, DIM + 1)
                    # assign dpos to trailing dim of E
                    for d in ti.static(range(DIM)):
                        E[d + 1] = dpos[d]
                    p_v_ws[p, adj_i - adj_v_idx_start] = (MInv @ (E * invD))[0]
                    # w_sum += p_v_ws[p, adj_i - adj_v_idx_start]
            #     print('w_sum is ', w_sum)
            # print('cal Dp below')
            # cal Dp
            tmp_Dp = ti.Matrix.zero(float, DIM, DIM)
            for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                v_idx = ti_f_1b_v_var[adj_i]
                w = p_v_ws[p, adj_i - adj_v_idx_start]
                dpos = ti_v_pos[v_idx] - x_p
                tmp_Dp += w * dpos.outer_product(dpos)
            # cache
            Dp_inv[p] = tmp_Dp.inverse()


@ti.kernel
def project_vertex_dirichelet():
    # hardcode dirichelet
    for v in ti_v_pos:
        tmp_pos = ti_v_pos[v]
        # check left set vel to 0
        if tmp_pos[0] < bc_dist_dx:
            ti_v_vel[v] *= 0


@ti.kernel
def particle_body_force():
    pass


@ti.kernel
def P2G():
    # P2G
    # particle to vertex
    for p in x:
        cell_idx = p_in_cell_idx[p]
        if cell_idx >= 0:
            tmp_Dp_inv = Dp_inv[p]
            # F[p]: deformation gradient update
            F[p] = (ti.Matrix.identity(float, DIM) + dt * C[p]) @ F[p]
            # h: Hardening coefficient: snow gets harder when compressed
            h = ti.exp(10 * (1.0 - Jp[p]))
            if material[p] == 1:  # elastic, donot change h
                h = 1.0
            mu, la = mu_0 * h, lambda_0 * h
            if material[p] == 0:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(F[p])
            J = 1.0
            I1 = 0.0
            for d in ti.static(range(DIM)):
                new_sig = sig[d, d]
                if material[p] == 2:  # Snow
                    new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
                Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
                I1 += new_sig**2
            if material[p] == 0:
                # Reset deformation gradient to avoid numerical instability
                F[p] = ti.Matrix.identity(float, DIM) * ti.sqrt(J)
            elif material[p] == 2:
                # Reconstruct elastic deformation gradient after plasticity
                F[p] = U @ sig @ V.transpose()
            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, DIM) * la * J * (J - 1)
            # post
            # theta_yy[p] = stress[1, 1]
            pot_eng[0] += (0.5 * mu * (I1 - DIM - 2 * ti.log(J)) + 0.5 * la * (J - 1)**2) * p_vol
            kin_eng[0] += 0.5 * p_mass * v[p].norm_sqr()
            gra_eng[0] += -p_mass * gravity * (x[p][1] - 6.0)
            # post
            stress = (-dt * p_vol) * stress @ tmp_Dp_inv
            affine = stress
            if ti.static(advect_scheme == ADV_TYPE.APIC):
                affine += p_mass * C[p]

            x_p = x[p]
            adj_v_idx_start = ti_f_1b_v_idx[cell_idx]
            adj_v_idx_end = ti_f_1b_v_idx[cell_idx + 1]
            for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                v_idx = ti_f_1b_v_var[adj_i]
                w = p_v_ws[p, adj_i - adj_v_idx_start]
                # vertex vars: mass, momentum(with stress and affine)
                ti_v_m[v_idx] += w * p_mass
                ti_v_vel_old[v_idx] += w * p_mass * v[p]
                ti_v_vel[v_idx] += w * (p_mass * v[p] + affine @ (ti_v_pos[v_idx] - x_p))


@ti.kernel
def body_force():
    # normalize the momentum to velocity at each vertex DOF
    # apply field force
    for v in ti_v_m:
        if ti_v_m[v] > 0:
            ti_v_vel[v] /= ti_v_m[v]
            ti_v_vel_old[v] /= ti_v_m[v]
            ti_v_vel[v][1] += dt * gravity  # gravity


@ti.kernel
def G2P():
    # particles from vertices
    for p in x:
        cell_idx = p_in_cell_idx[p]
        x_p = x[p]
        tmp_Dp_inv = Dp_inv[p]

        new_x = ti.Vector.zero(float, DIM)
        new_v = ti.Vector.zero(float, DIM)
        delta_v = ti.Vector.zero(float, DIM)
        new_B = ti.Matrix.zero(float, DIM, DIM)

        adj_v_idx_start = ti_f_1b_v_idx[cell_idx]
        adj_v_idx_end = ti_f_1b_v_idx[cell_idx + 1]
        for adj_i in range(adj_v_idx_start, adj_v_idx_end):
            v_idx = ti_f_1b_v_var[adj_i]
            w = p_v_ws[p, adj_i - adj_v_idx_start]

            new_x += w * (ti_v_pos[v_idx] + dt * ti_v_vel[v_idx])
            new_v += w * ti_v_vel[v_idx]
            delta_v += w * (ti_v_vel[v_idx] - ti_v_vel_old[v_idx])
            new_B += w * (ti_v_vel[v_idx].outer_product(ti_v_pos[v_idx] - x_p))

        if ti.static(advect_scheme == ADV_TYPE.FLIP):
            v[p] = v[p] + delta_v
        else:
            v[p] = new_v
        B[p] = new_B
        C[p] = new_B @ tmp_Dp_inv
        x[p] = new_x


@ti.kernel
def sum_p_momentum():
    L_A_p[0] *= 0
    for p in x:
        m = p_mass
        vel = v[p]
        pos = x[p]

        L_A_p[0][0] += m * vel[0]
        L_A_p[0][1] += m * vel[1]
        L_A_p[0][2] += m * pos.cross(vel) + m * (B[p][1, 0] - B[p][0, 1])


@ti.kernel
def sum_g_momentum():
    L_A_g[0] *= 0
    # instead of looping grids, now looping vertices
    # then accumulate the momentum and angular momentum
    for v in ti_v_m:
        m = ti_v_m[v]
        if m > 0:
            vel = ti_v_vel[v]
            pos = ti_v_pos[v]

            L_A_g[0][0] += m * vel[0]
            L_A_g[0][1] += m * vel[1]
            L_A_g[0][2] += m * pos.cross(vel)


@ti.kernel
def push_back_into_mesh():
    # iterate all particles' position
    # clamp it into the box size
    for p in x:
        # push in x direction
        if x[p][0] < 0 + bc_dist_dx:
            x[p][0] = 0 + bc_dist_dx
        elif x[p][0] > Lx - bc_dist_dx:
            x[p][0] = Lx - bc_dist_dx
        # push in y direction
        if x[p][1] < 0 + bc_dist_dy:
            x[p][1] = 0 + bc_dist_dy
        elif x[p][1] > Ly - bc_dist_dy:
            x[p][1] = Ly - bc_dist_dy


def substep(t):
    # zero init containers
    zero_clear()
    if verbose:
        print('after init zero clear')
        print('max particle velocity: ', np.max(np.abs(v.to_numpy())))
        print('max vertex velocity: ', np.max(np.abs(ti_v_vel.to_numpy())))

    # call back
    call_back(t)

    # DP_inv cal
    cal_DP_inv()
    if verbose:
        sum_p_momentum()
        print('part momentum before p2g: ', L_A_p[0][0], L_A_p[0][1])
        print('part angular momentum before p2g: ', L_A_p[0][2])

    if verbose:
        print('after DP inv')
        print('max particle velocity: ', np.max(np.abs(v.to_numpy())))
        print('max vertex velocity: ', np.max(np.abs(ti_v_vel.to_numpy())))

        print('max particle F coeff: ', np.max(np.abs(F.to_numpy())))
        print('max particle C coeff: ', np.max(np.abs(C.to_numpy())))
        print('max particle Jp coeff: ', np.max(np.abs(Jp.to_numpy())))

    # # particle body force
    # particle_body_force()

    # P2G
    P2G()
    if verbose:
        # TODO now the grid update and P2G is combined
        # TODO furthur test by splitting them
        #
        sum_g_momentum()
        print('grid momentum after  p2g: ', L_A_g[0][0], L_A_g[0][1])
        print('grid angular momentum after  p2g: ', L_A_g[0][2])

    if verbose:
        # check particles, vertices, cells maximum absolute velocity
        print('after p2g1')
        print('max particle velocity: ', np.max(np.abs(v.to_numpy())))
        print('max vertex velocity: ', np.max(np.abs(ti_v_vel.to_numpy())))
        print('max particle F coeff: ', np.max(np.abs(F.to_numpy())))
        print('max particle C coeff: ', np.max(np.abs(C.to_numpy())))
        print('max particle Jp coeff: ', np.max(np.abs(Jp.to_numpy())))

    # body force
    body_force()

    # cell dirichelet
    project_vertex_dirichelet()
    if verbose:
        print('after cell dirichelet')
        print('max particle velocity: ', np.max(np.abs(v.to_numpy())))
        print('max vertex velocity: ', np.max(np.abs(ti_v_vel.to_numpy())))

    # G2P
    G2P()
    if verbose:
        sum_p_momentum()
        print('part momentum after  g2p: ', L_A_p[0][0], L_A_p[0][1])
        print('part angular momentum after g2p: ', L_A_p[0][2])

    if verbose:
        print('after g2p')
        print('max particle velocity: ', np.max(np.abs(v.to_numpy())))
        print('max vertex velocity: ', np.max(np.abs(ti_v_vel.to_numpy())))

    # safety clamp
    # NOTE TODO have to come up with a general solution for any mesh
    push_back_into_mesh()


def main():
    # title
    dump_path = f'{DIM}d_cantilever_weaker'
    dump_path += '_' + f'quality{quality}'
    dump_path += '_' + ADV_TYPE_STR[advect_scheme.value]

    init_mesh()
    # for ti visulaizeion
    line = mesh2line(ti_v_pos.to_numpy(), ti_cell.to_numpy())
    zoom_ratio = max(Lx, Ly)
    line /= zoom_ratio
    gui = ti.GUI(dump_path, res=512, background_color=0x112F41, show_gui=False)
    # for ply
    mesh_cell = meshio.Mesh(points=ti_v_pos.to_numpy(), cells={'triangle': ti_cell.to_numpy()})
    dump_path_cell = os.path.join(dump_path, 'cell.vtk')
    os.makedirs(dump_path, exist_ok=True)
    mesh_cell.write(dump_path_cell)

    frame = 0
    # post script
    sample_p_disy_log = np.zeros((end_frame + 1, 1))
    kin_eng_log = np.zeros((end_frame + 1, 1))
    pot_eng_log = np.zeros((end_frame + 1, 1))
    p_gra_eng_log = np.zeros((end_frame + 1, 1))
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        print(f'frame {frame}')

        # pre script
        # sample trajectory (pos) vel and eng of sample_idx
        sample_p_disy_log[frame] = x[sample_idx][1] - sampled_pos[1]
        # pre script

        step_per_frame = int(frame_dt // dt)
        for s in range(step_per_frame):
            substep(frame * frame_dt + s * dt)

        # ti visualization
        # zoom ratio is the max of Lx or Ly

        gui.circles(x.to_numpy() / zoom_ratio, radius=1.5, palette=[0x068587, 0xED553B, 0xEEEEF0], palette_indices=material)
        gui.lines(line[:, 0, :2], line[:, 1, :2], color=0x000064)
        if output_png:
            gui.show(os.path.join(dump_path, f'frame{frame:d}.png'))
        else:
            gui.show()
        # dump vtk
        if output_vtk:
            mesh_p = meshio.Mesh(points=x.to_numpy(), cells={})
            dump_path_p = os.path.join(dump_path, f'particle{frame}.vtk')
            mesh_p.write(dump_path_p)

        # post script
        # post script
        kin_eng_log[frame] = kin_eng[0]
        pot_eng_log[frame] = pot_eng[0]
        p_gra_eng_log[frame] = gra_eng[0]
        # post script

        frame += 1

        if frame > end_frame:
            break

    # plot
    paint_res = 1080

    # middle point dis y
    # clean plt
    plt.clf()
    # set resolution
    plt.rcParams['figure.dpi'] = paint_res
    # set font
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel(r'$t[s]$')
    plt.ylabel(r'$u_y[m]$')
    # plt.xlim(0, 3)
    # plt.ylim(-3.5, 0)
    # set legend
    legend = [r'$ours$']
    # time series: [0:end_frame] * frame_dt
    plt.plot(np.arange(0, end_frame + 1) * frame_dt, sample_p_disy_log, label=legend[0])
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(dump_path, f'sample_p_disy_log.png'), dpi=paint_res)
    np.save(os.path.join(dump_path, f'sample_p_disy_log.npy'), sample_p_disy_log)

    # total energy
    # clean plt
    plt.clf()
    # plot energies
    plt.rcParams['figure.dpi'] = paint_res
    # set font
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel(r'$t[s]$')
    plt.ylabel(r'$E[J]$')
    # set the unit of y axis to be 10^(-3)
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(-7, -7))
    # plt.xlim(0, 2.5)
    # plt.ylim(-75, 75)
    # set legend
    legend = [r'$E_{kin}$', r'$E_{pot}$', r'$E_{grav}$', r'$E_{tot}$']
    # plot kin, pot, and tot vs time series: [0:end_frame] * frame_dt
    tot_eng_log = kin_eng_log + pot_eng_log + p_gra_eng_log
    plt.plot(np.arange(0, end_frame + 1) * frame_dt, kin_eng_log, label=legend[0])
    plt.plot(np.arange(0, end_frame + 1) * frame_dt, pot_eng_log, label=legend[1])
    plt.plot(np.arange(0, end_frame + 1) * frame_dt, p_gra_eng_log, label=legend[2])
    plt.plot(np.arange(0, end_frame + 1) * frame_dt, tot_eng_log, label=legend[3])
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(dump_path, f'energies.png'), dpi=paint_res)
    np.savez(os.path.join(dump_path, f'energies.npz'), {'pot': pot_eng_log, 'kin': kin_eng_log, 'gra': p_gra_eng_log, 'tot': tot_eng_log})


if __name__ == '__main__':
    main()