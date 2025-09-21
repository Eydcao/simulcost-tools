import taichi as ti
import numpy as np
import os
from ..helpers import barycentric_coord_2d, barycentric_coord_3d, B_spline, ADV_TYPE
from ..CTRL_DATAs.base import CTRL_DATA
import meshio


@ti.data_oriented
class MPM_FIELD_BASE:
    def __init__(self, ctrl_data):
        self.sum_each_part_neighbor_communication = ti.field(dtype=float, shape=1)
        # singleton ctrl data
        assert isinstance(ctrl_data, CTRL_DATA)
        self.ctrl_data = ctrl_data

        # particle attributes
        self.x = ti.Vector.field(self.ctrl_data.DIM, dtype=float, shape=self.ctrl_data.n_p)  # position
        self.x_old = ti.Vector.field(self.ctrl_data.DIM, dtype=float, shape=self.ctrl_data.n_p)  # position old
        self.v = ti.Vector.field(self.ctrl_data.DIM, dtype=float, shape=self.ctrl_data.n_p)  # velocity
        self.v_old = ti.Vector.field(self.ctrl_data.DIM, dtype=float, shape=self.ctrl_data.n_p)  # velocity old
        self.C = ti.Matrix.field(
            self.ctrl_data.DIM,
            self.ctrl_data.DIM,
            dtype=float,
            shape=self.ctrl_data.n_p,
        )  # affine velocity field
        self.B = ti.Matrix.field(
            self.ctrl_data.DIM,
            self.ctrl_data.DIM,
            dtype=float,
            shape=self.ctrl_data.n_p,
        )
        self.F = ti.Matrix.field(
            self.ctrl_data.DIM,
            self.ctrl_data.DIM,
            dtype=float,
            shape=self.ctrl_data.n_p,
        )  # deformation gradient
        self.sigma = ti.Matrix.field(
            self.ctrl_data.DIM,
            self.ctrl_data.DIM,
            dtype=float,
            shape=self.ctrl_data.n_p,
        )  # stress
        self.sig_out = ti.Vector.field(self.ctrl_data.DIM, dtype=float, shape=self.ctrl_data.n_p)  # for isotropic
        self.material = ti.field(dtype=int, shape=self.ctrl_data.n_p)  # material id
        self.Jp = ti.field(dtype=float, shape=self.ctrl_data.n_p)  # plastic deformation
        self.Dp_inv = ti.Matrix.field(
            self.ctrl_data.DIM,
            self.ctrl_data.DIM,
            dtype=float,
            shape=self.ctrl_data.n_p,
        )
        # self.Dp_inv = ti.Matrix.field(self.ctrl_data.DIM, self.ctrl_data.DIM, dtype=ti.f64, shape=self.ctrl_data.n_p)
        # p_in_g information
        # N0 is the zero ring, ie the cell
        # N1 is the 1 ring neighbors of N0
        self.p_in_cell_idx = ti.field(dtype=int, shape=self.ctrl_data.n_p)
        self.p_in_cell_bcoord = ti.Vector.field(
            self.ctrl_data.DIM + 1, dtype=float, shape=self.ctrl_data.n_p
        )  # barycentric coord
        # p_v_ws's shape is (n_v, self.ctrl_data.max_nonzero_N1_v), as a particle can only appear in one N0 and interact with all of its 1b adj vertices
        # self.p_v_ws = ti.field(dtype=ti.f64, shape=(self.ctrl_data.n_p, self.ctrl_data.max_nonzero_N1_v))
        self.p_v_ws = ti.field(dtype=float, shape=(self.ctrl_data.n_p, self.ctrl_data.max_nonzero_N1_v))
        self.p_support_radii = ti.field(dtype=float, shape=self.ctrl_data.n_p)

        # static connectivity
        self.N0 = ti.Vector.field(self.ctrl_data.DIM + 1, dtype=int, shape=self.ctrl_data.n_cell)
        self.N1_v_idx = ti.field(dtype=int, shape=self.ctrl_data.N1_v_idx.shape[0])
        self.N1_v_var = ti.field(dtype=int, shape=self.ctrl_data.N1_v_var.shape[0])
        self.adj_N1_v_N0_v = ti.Vector.field(
            self.ctrl_data.DIM + 1,
            dtype=float,
            shape=self.ctrl_data.adj_N1_v_N0_v.shape[0],
        )

        # v attributes: pos, v3l, m, vel, dirichelet flag (TODO set dirichelet flag)
        # static
        self.v_pos = ti.Vector.field(self.ctrl_data.DIM, dtype=float, shape=self.ctrl_data.n_v)
        # self.v_dirichlet = ti.field(dtype=int, shape=self.ctrl_data.n_v)
        self.v_is_bc = ti.field(dtype=int, shape=self.ctrl_data.n_v)
        self.v_outgoing_normal = ti.Vector.field(self.ctrl_data.DIM, dtype=float, shape=self.ctrl_data.n_v)
        self.v_support_radii = ti.field(dtype=float, shape=self.ctrl_data.n_v)
        # dynamic
        self.v_m = ti.field(dtype=float, shape=self.ctrl_data.n_v)
        self.v_vel = ti.Vector.field(self.ctrl_data.DIM, dtype=float, shape=self.ctrl_data.n_v)
        self.v_vel_old = ti.Vector.field(
            self.ctrl_data.DIM, dtype=float, shape=self.ctrl_data.n_v
        )  # can be either v_new or delta_v; only used in flip

        # hash map attributes
        self.hash2N0_idx = ti.field(dtype=int, shape=self.ctrl_data.hash2N0_idx.shape[0])
        self.hash2N0_var = ti.field(dtype=int, shape=self.ctrl_data.hash2N0_var.shape[0])

        # summed vars
        self.L_A_p = ti.Vector.field(self.ctrl_data.DIM + 1, dtype=float, shape=1)
        self.L_A_g = ti.Vector.field(self.ctrl_data.DIM + 1, dtype=float, shape=1)
        self.pot_eng = ti.field(dtype=float, shape=1)
        self.kin_eng = ti.field(dtype=float, shape=1)
        self.gra_eng = ti.field(dtype=float, shape=1)

        # deactivate flag
        self.deactivated_flag = ti.field(dtype=int, shape=self.ctrl_data.n_p)

        # post script fields
        self.declare_additional_fields()

    def declare_additional_fields(self):
        # TODO implement in override
        pass

    def init_fields(self):
        # init particle attributes
        self.x.from_numpy(self.ctrl_data.p_pos.astype(np.float32))
        self.material.from_numpy(self.ctrl_data.p_mat)
        self.v.from_numpy(self.ctrl_data.p_vel.astype(np.float32))
        self.F.from_numpy(self.ctrl_data.p_F.astype(np.float32))
        self.Jp.from_numpy(self.ctrl_data.p_Jp.astype(np.float32))
        # print("init particle attributes done")
        # init v attributes
        self.v_pos.from_numpy(self.ctrl_data.v_pos.astype(np.float32))
        self.v_is_bc.from_numpy(self.ctrl_data.is_bc)
        self.v_outgoing_normal.from_numpy(self.ctrl_data.outgoing_normal.astype(np.float32))
        self.v_support_radii.from_numpy(self.ctrl_data.v_support_radii.astype(np.float32))
        # print("init v attributes done")
        # init cell connectivity
        self.N0.from_numpy(self.ctrl_data.cell)
        self.N1_v_idx.from_numpy(self.ctrl_data.N1_v_idx)
        self.N1_v_var.from_numpy(self.ctrl_data.N1_v_var)
        self.adj_N1_v_N0_v.from_numpy(self.ctrl_data.adj_N1_v_N0_v.astype(np.float32))
        # print("init cell connectivity done")
        # init hash map
        self.hash2N0_idx.from_numpy(self.ctrl_data.hash2N0_idx)
        self.hash2N0_var.from_numpy(self.ctrl_data.hash2N0_var)
        # print("init hash map done")
        # set_dirichelet TODO
        # init active flag; defualt all zero
        self.deactivated_flag.from_numpy(np.zeros(self.ctrl_data.n_p, dtype=int))
        # init communication counter
        self.sum_each_part_neighbor_communication.fill(0.0)
        # print("init active flag done")

    @ti.func
    def hashing(self, spatial_idx: ti.template()):
        # clamp res into hash_min and hash_max
        ti_hash_max = ti.Vector(self.ctrl_data.hash_max)
        ti_hash_min = ti.Vector(self.ctrl_data.hash_min)
        spatial_idx = ti.min(ti.max(spatial_idx, ti_hash_min), ti_hash_max)
        res = (spatial_idx - ti_hash_min) * ti.Vector(self.ctrl_data.hash_stride)
        ws = ti.Vector.one(int, self.ctrl_data.DIM)

        return res.dot(ws)

    @ti.kernel
    def zero_clear_additional_fields(self):
        # TODO implement in override
        pass

    @ti.kernel
    def zero_clear(self):
        # zero all elements in v containers
        for v in self.v_m:
            self.v_m[v] *= 0
            self.v_vel_old[v] *= 0
            self.v_vel[v] *= 0
        # zero all p_v_ws
        for p, b in self.p_v_ws:
            self.p_v_ws[p, b] = 0

        # cleared summed vars
        self.L_A_p[0] *= 0
        self.L_A_g[0] *= 0
        self.pot_eng[0] *= 0
        self.kin_eng[0] *= 0
        self.gra_eng[0] *= 0

    @ti.kernel
    def call_back(self, t: float):
        # implement in override
        pass

    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

    @ti.kernel
    def cal_DP_inv(self) -> bool: # TODO: sum_N_adj
        # Dp_inv calculation
        # NOTE BTW, calculate p_in_mesh informations, re-use them in later g2p or p2g processes
        any_p_not_in_mesh = False
        for p in self.x:
            if self.deactivated_flag[p] == 1:
                continue
            # force search
            found_cell = False
            # find which hash box the particle is in
            hash_vec = ti.Vector(self.ctrl_data.hash_dx)
            hash_spatial_idx = ti.floor(self.x[p] / hash_vec, dtype=int)
            hash_idx = self.hashing(hash_spatial_idx)
            # only iterate through the cells inters with this hash box
            inter_cell_start = self.hash2N0_idx[hash_idx]
            inter_cell_end = self.hash2N0_idx[hash_idx + 1]
            for tidx in range(inter_cell_start, inter_cell_end):
                f = self.hash2N0_var[tidx]
                v0 = self.v_pos[self.N0[f][0]]
                v1 = self.v_pos[self.N0[f][1]]
                v2 = self.v_pos[self.N0[f][2]]
                v3 = ti.Vector.zero(float, self.ctrl_data.DIM)
                if ti.static(self.ctrl_data.DIM == 3):
                    v3 = self.v_pos[self.N0[f][3]]
                P = self.x[p]
                # get the max and min of the v0 to v3
                # use bbox to filter
                min_bb = ti.Vector.zero(float, self.ctrl_data.DIM)
                max_bb = ti.Vector.zero(float, self.ctrl_data.DIM)
                if ti.static(self.ctrl_data.DIM == 2):
                    min_bb = ti.min(v0, v1, v2)
                    max_bb = ti.max(v0, v1, v2)
                elif ti.static(self.ctrl_data.DIM == 3):
                    min_bb = ti.min(v0, v1, v2, v3)
                    max_bb = ti.max(v0, v1, v2, v3)
                if all(P >= min_bb - self.ctrl_data.eps) and all(P <= max_bb + self.ctrl_data.eps):
                    # get the b_coord of this p in this cell
                    # if all larger than zero, then this p is in this cell
                    b_coord = ti.Vector.zero(float, self.ctrl_data.DIM + 1)
                    if ti.static(self.ctrl_data.DIM == 2):
                        b_coord = barycentric_coord_2d(v0, v1, v2, P)
                    elif ti.static(self.ctrl_data.DIM == 3):
                        b_coord = barycentric_coord_3d(v0, v1, v2, v3, P)
                    if all(b_coord >= -self.ctrl_data.eps):
                        # clamp b_coord all to between 0 and 1 to avoid errors
                        b_coord = ti.max(
                            ti.min(b_coord, ti.Vector.one(float, self.ctrl_data.DIM + 1)),
                            ti.Vector.zero(float, self.ctrl_data.DIM + 1),
                        )
                        # normalize
                        b_coord /= b_coord.sum()
                        # print('b_coord is ', b_coord)
                        # adjust p pos to correspond with b_coord
                        # self.x[p] = v0 * b_coord[0] + v1 * b_coord[1] + v2 * b_coord[2]
                        # if ti.static(self.ctrl_data.DIM == 3):
                        #     self.x[p] += v3 * b_coord[3]
                        self.p_in_cell_idx[p] = f
                        self.p_in_cell_bcoord[p] = b_coord
                        # assign p support radii
                        v_rs = ti.Vector.zero(float, self.ctrl_data.DIM + 1)
                        for d in ti.static(range(self.ctrl_data.DIM + 1)):
                            v_rs[d] = self.v_support_radii[self.N0[f][d]]
                        self.p_support_radii[p] = v_rs.dot(b_coord)
                        found_cell = True
                        break
            if not found_cell:
                # print('error not found cell')
                # print('p, p_pos ', p, self.x[p])
                # exit(1)
                # any_p_not_in_mesh = True
                # self.p_in_cell_idx[p] = -1
                # self.p_support_radii[p] = 0.0
                # self.deactivated_flag[p] = 1

                # restore the old x, not changing p_in_cell_idx p_in_cell_bcoord nor p_support_radii
                # direction_adv = self.x[p] - self.x_old[p]
                # dir_norm = self.norm(direction_adv)
                # direction_adv = direction_adv / dir_norm
                self.x[p] = self.x_old[p]
                # project the velocity on the direction_adv
                # v_dir_norm = self.v_old[p].dot(direction_adv)
                # v_dir = direction_adv * v_dir_norm
                # v_remain = self.v_old[p] - v_dir
                self.v[p] *= 0
            else:
                # get which cell this p is in
                cell_idx = self.p_in_cell_idx[p]
                b_coord = self.p_in_cell_bcoord[p]
                # calculate the p_v_ws using moving least square on the fly
                x_p = self.x[p]
                adj_v_idx_start = self.N1_v_idx[cell_idx]
                adj_v_idx_end = self.N1_v_idx[cell_idx + 1]
                N_adj = adj_v_idx_end - adj_v_idx_start
                # edge case 1: smaller than dim+1 adj vertices, try uniform weights
                # NOTE but this case should never (precondition) happen because it degenerates
                if adj_v_idx_end - adj_v_idx_start < self.ctrl_data.DIM + 1:
                    self.sum_each_part_neighbor_communication[0] += N_adj
                    for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                        self.p_v_ws[p, adj_i - adj_v_idx_start] = 1.0 / (adj_v_idx_end - adj_v_idx_start)
                # edge case 2: if equal to DIM + 1, use barrycentric coord
                # NOTE this case only happens when there is a single element
                elif adj_v_idx_end - adj_v_idx_start == self.ctrl_data.DIM + 1:
                    self.sum_each_part_neighbor_communication[0] += N_adj
                    for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                        self.p_v_ws[p, adj_i - adj_v_idx_start] = b_coord[adj_i - adj_v_idx_start]
                else:
                    # calculate the M = ETDInv @ E by summing up the dpos/D outproduct with dpos
                    # since taichi doesnot support dynamic matrix
                    M = ti.Matrix.zero(ti.f32, self.ctrl_data.DIM + 1, self.ctrl_data.DIM + 1)
                    # support_radii = self.p_support_radii[p] / 2.0  # NOTE should it be 2 or 1.5
                    support_radii = self.ctrl_data.dx * 2.0
                    # print('support_radii is ', support_radii)
                    self.sum_each_part_neighbor_communication[0] += N_adj * self.ctrl_data.DIM
                    for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                        v_idx = self.N1_v_var[adj_i]
                        v_connect_2_cell_v = self.adj_N1_v_N0_v[adj_i]
                        dpos = self.v_pos[v_idx] - x_p
                        d_hat = dpos.norm() / support_radii
                        invD = B_spline(d_hat)
                        # get the sum of product of v_connect_2_cell_v and b_coord
                        temp = v_connect_2_cell_v * b_coord
                        s = temp.sum()
                        # diminish invD by s
                        invD *= s
                        invD = ti.max(invD, self.ctrl_data.eps)
                        E = ti.Vector.one(float, self.ctrl_data.DIM + 1)
                        # assign dpos to trailing dim of E
                        for d in ti.static(range(self.ctrl_data.DIM)):
                            E[d + 1] = dpos[d]
                        M += (E * invD).outer_product(E)
                    MInv = M.inverse()
                    # calculate the p_v_ws = MInv @ ETDInv's col(i) by iterating through all adj vertices
                    # since taichi doesnot support dynamic matrix
                    # w_sum = 0.0
                    self.sum_each_part_neighbor_communication[0] += N_adj * self.ctrl_data.DIM
                    for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                        v_idx = self.N1_v_var[adj_i]
                        v_connect_2_cell_v = self.adj_N1_v_N0_v[adj_i]
                        dpos = self.v_pos[v_idx] - x_p
                        # sepeartion factor D
                        d_hat = dpos.norm() / support_radii
                        invD = B_spline(d_hat)
                        # get the sum of product of v_connect_2_cell_v and b_coord
                        temp = v_connect_2_cell_v * b_coord
                        s = temp.sum()
                        # diminish invD by s
                        invD *= s
                        invD = ti.max(invD, self.ctrl_data.eps)
                        E = ti.Vector.one(float, self.ctrl_data.DIM + 1)
                        # assign dpos to trailing dim of E
                        for d in ti.static(range(self.ctrl_data.DIM)):
                            E[d + 1] = dpos[d]
                        self.p_v_ws[p, adj_i - adj_v_idx_start] = (MInv @ (E * invD))[0]
                        # w_sum += p_v_ws[p, adj_i - adj_v_idx_start]
                #     print('w_sum is ', w_sum)
                # print('cal Dp below')
                # cal Dp
                tmp_Dp = ti.Matrix.zero(ti.f32, self.ctrl_data.DIM, self.ctrl_data.DIM)
                self.sum_each_part_neighbor_communication[0] += N_adj
                for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                    v_idx = self.N1_v_var[adj_i]
                    w = self.p_v_ws[p, adj_i - adj_v_idx_start]
                    dpos = self.v_pos[v_idx] - x_p
                    tmp_Dp += w * dpos.outer_product(dpos)
                # cache
                self.Dp_inv[p] = tmp_Dp.inverse()

        return any_p_not_in_mesh

    @ti.kernel
    def project_vertex_dirichelet(self):
        # NOTE: by defualt only use sliding boundary on bc nodes
        #       if you want DIY, override this function
        # for v in self.v_pos:
        #     if self.v_is_bc[v]:
        #         node_out_normal = self.v_outgoing_normal[v]

        #         vel_old_normal = self.v_vel_old[v].dot(node_out_normal)
        #         vel_normal = self.v_vel[v].dot(node_out_normal)
        #         self.v_vel_old[v] -= vel_old_normal * node_out_normal
        #         self.v_vel[v] -= vel_normal * node_out_normal
        pass

    @ti.kernel
    def particle_body_force(self):
        # TODO implement in overide
        pass

    @ti.kernel
    def particle_penalty_force(self, t: float):
        pass

    @ti.kernel
    def P2G(self):
        # P2G
        # particle to vertex
        for p in self.x:
            cell_idx = self.p_in_cell_idx[p]
            if cell_idx >= 0:
                tmp_Dp_inv = self.Dp_inv[p]
                # F[p]: deformation gradient update
                self.F[p] = (ti.Matrix.identity(float, self.ctrl_data.DIM) + self.ctrl_data.dt * self.C[p]) @ self.F[p]
                # h: Hardening coefficient: snow gets harder when compressed
                h = ti.exp(10 * (1.0 - self.Jp[p]))
                if self.material[p] == 1:  # elastic, donot change h
                    h = 1.0
                mu, la = self.ctrl_data.mu_0 * h, self.ctrl_data.lambda_0 * h
                if self.material[p] == 0:  # liquid
                    mu = 0.0
                U, sig, V = ti.svd(self.F[p].cast(ti.f32))
                J = 1.0
                I1 = 0.0
                for d in ti.static(range(self.ctrl_data.DIM)):
                    new_sig = sig[d, d]
                    if self.material[p] == 2:  # Snow
                        new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
                    self.Jp[p] *= sig[d, d] / new_sig
                    sig[d, d] = new_sig
                    J *= new_sig
                    I1 += new_sig**2
                if self.material[p] == 0:
                    # Reset deformation gradient to avoid numerical instability
                    self.F[p] = ti.Matrix.identity(float, self.ctrl_data.DIM) * ti.sqrt(J)
                elif self.material[p] == 2:
                    # Reconstruct elastic deformation gradient after plasticity
                    self.F[p] = U @ sig @ V.transpose()
                stress = 2 * mu * (self.F[p].cast(ti.f32) - U @ V.transpose()) @ self.F[p].cast(
                    ti.f32
                ).transpose() + ti.Matrix.identity(float, self.ctrl_data.DIM) * la * J * (J - 1)

                # record stress
                self.sigma[p] = stress
                # record energy
                self.pot_eng[0] += (
                    0.5 * mu * (I1 - self.ctrl_data.DIM - 2 * ti.log(J)) + 0.5 * la * (J - 1) ** 2
                ) * self.ctrl_data.p_vol
                self.kin_eng[0] += 0.5 * self.ctrl_data.p_mass * self.v[p].norm_sqr()
                self.gra_eng[0] += -self.ctrl_data.p_mass * self.ctrl_data.gravity * self.x[p][1]

                stress = (-self.ctrl_data.dt * self.ctrl_data.p_vol) * stress @ tmp_Dp_inv
                affine = stress
                if ti.static(self.ctrl_data.advect_scheme == ADV_TYPE.APIC):
                    affine += self.ctrl_data.p_mass * self.C[p]

                x_p = self.x[p]
                adj_v_idx_start = self.N1_v_idx[cell_idx]
                adj_v_idx_end = self.N1_v_idx[cell_idx + 1]
                N_adj = adj_v_idx_end - adj_v_idx_start
                self.sum_each_part_neighbor_communication[0] += N_adj
                for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                    v_idx = self.N1_v_var[adj_i]
                    w = self.p_v_ws[p, adj_i - adj_v_idx_start]

                    self.v_m[v_idx] += w * self.ctrl_data.p_mass
                    self.v_vel_old[v_idx] += w * self.ctrl_data.p_mass * self.v[p]
                    self.v_vel[v_idx] += w * (self.ctrl_data.p_mass * self.v[p] + affine @ (self.v_pos[v_idx] - x_p))

    @ti.kernel
    def body_force(self):
        # normalize the momentum to velocity at each vertex DOF
        # apply field force
        for v in self.v_m:
            if self.v_m[v] > 0:
                self.v_vel[v] /= self.v_m[v]
                self.v_vel_old[v] /= self.v_m[v]
                self.v_vel[v][1] += self.ctrl_data.dt * self.ctrl_data.gravity  # gravity

    @ti.kernel
    def G2P(self):
        # particles from vertices
        for p in self.x:
            cell_idx = self.p_in_cell_idx[p]
            x_p = self.x[p]
            self.x_old[p] = x_p  # backup old pos
            self.v_old[p] = self.v[p]  # backup old vel
            tmp_Dp_inv = self.Dp_inv[p]

            new_x = ti.Vector.zero(float, self.ctrl_data.DIM)
            new_v = ti.Vector.zero(float, self.ctrl_data.DIM)
            delta_v = ti.Vector.zero(float, self.ctrl_data.DIM)
            new_B = ti.Matrix.zero(float, self.ctrl_data.DIM, self.ctrl_data.DIM)

            adj_v_idx_start = self.N1_v_idx[cell_idx]
            adj_v_idx_end = self.N1_v_idx[cell_idx + 1]
            N_adj = adj_v_idx_end - adj_v_idx_start
            self.sum_each_part_neighbor_communication[0] += N_adj
            for adj_i in range(adj_v_idx_start, adj_v_idx_end):
                v_idx = self.N1_v_var[adj_i]
                w = self.p_v_ws[p, adj_i - adj_v_idx_start]

                new_x += w * (self.v_pos[v_idx] + self.ctrl_data.dt * self.v_vel[v_idx])
                new_v += w * self.v_vel[v_idx]
                delta_v += w * (self.v_vel[v_idx] - self.v_vel_old[v_idx])
                new_B += w * (self.v_vel[v_idx].outer_product(self.v_pos[v_idx] - x_p))

            if ti.static(self.ctrl_data.advect_scheme == ADV_TYPE.FLIP):
                ratio = self.ctrl_data.flip_ratio
                self.v[p] = (self.v[p] + delta_v) * ratio + new_v * (1 - ratio)
            else:
                self.v[p] = new_v
            self.B[p] = new_B
            self.C[p] = new_B @ tmp_Dp_inv
            self.x[p] = new_x

    @ti.kernel
    def sum_p_momentum(self):
        self.L_A_p[0] *= 0
        for p in self.x:
            m = self.ctrl_data.p_mass
            vel = self.v[p]
            pos = self.x[p]

            self.L_A_p[0][0] += m * vel[0]
            self.L_A_p[0][1] += m * vel[1]
            self.L_A_p[0][2] += m * pos.cross(vel) + m * (self.B[p][1, 0] - self.B[p][0, 1])

    @ti.kernel
    def sum_g_momentum(self):
        self.L_A_g[0] *= 0
        # instead of looping grids, now looping vertices
        # then accumulate the momentum and angular momentum
        # TODO only works for 2D; 3D tobe implemented
        if ti.static(self.ctrl_data.DIM == 2):
            for v in self.v_m:
                m = self.v_m[v]
                if m > 0:
                    vel = self.v_vel[v]
                    pos = self.v_pos[v]

                    self.L_A_g[0][0] += m * vel[0]
                    self.L_A_g[0][1] += m * vel[1]
                    self.L_A_g[0][2] += m * pos.cross(vel)
        elif ti.static(self.ctrl_data.DIM == 3):
            raise NotImplementedError
        else:
            raise NotImplementedError

    @ti.kernel
    def push_back_into_mesh(self):
        # TODO implement in override
        # TODO also consider a general approach for any mesh
        pass

    @ti.kernel
    def setup_collapse_flag(self):
        pass
        # for p in self.x:
        #     if self.x[p][1]>-0.24:
        #         self.collapse_flag[p] = 1

    def substep(self, t: float):
        if t <= 2 * self.ctrl_data.dt:
            self.setup_collapse_flag()

        # zero init containers
        self.zero_clear()
        # print('after zero_clear')
        self.zero_clear_additional_fields()
        # print('after zero_clear_additional_fields')

        # call back
        self.call_back(t)
        # print('after call_back')

        # DP_inv cal
        any_p_not_in_cell = self.cal_DP_inv()
        # print('after cal_DP_inv')

        # if any_p_not_in_cell:
        #     print('any p not in mesh: ', any_p_not_in_cell)
        #     # exit(1)

        if self.ctrl_data.verbose:
            self.sum_p_momentum()
            print("part momentum before p2g: ", self.L_A_p[0][0], self.L_A_p[0][1])
            print("part angular momentum before p2g: ", self.L_A_p[0][2])

        # particle body force
        self.particle_body_force()
        # print('after particle_body_force')

        # particle penalty force
        self.particle_penalty_force(t)
        # print('after particle_penalty_force')

        # P2G
        self.P2G()
        # print('after P2G')

        # body force
        self.body_force()
        # print('after body_force')

        if self.ctrl_data.verbose:
            self.sum_g_momentum()
            print("grid momentum after  p2g: ", self.L_A_g[0][0], self.L_A_g[0][1])
            print("grid angular momentum after  p2g: ", self.L_A_g[0][2])

        # cell dirichelet
        self.project_vertex_dirichelet()
        # print('after project_vertex_dirichelet')

        # G2P
        self.G2P()
        # print('after G2P')
        if self.ctrl_data.verbose:
            self.sum_p_momentum()
            print("part momentum after g2p: ", self.L_A_p[0][0], self.L_A_p[0][1])
            print("part angular momentum after g2p: ", self.L_A_p[0][2])

        # safety clamp
        self.push_back_into_mesh()
        # print('after push_back_into_mesh')

    def dump_mesh(self):
        cell_key = "triangle" if self.ctrl_data.DIM == 2 else "tetra"
        mesh_cell = meshio.Mesh(
            points=self.ctrl_data.v_pos,
            cells={cell_key: self.ctrl_data.cell},
            point_data={"is_bc": self.ctrl_data.is_bc.astype(float)},
        )
        dump_path_cell = os.path.join(self.ctrl_data.cache_dir, "res", "background_cell.vtk")
        os.makedirs(os.path.join(self.ctrl_data.cache_dir, "res"), exist_ok=True)
        mesh_cell.write(dump_path_cell)

    def dump(self, frame):
        F_np = self.F.to_numpy()
        Sigma_np = self.sigma.to_numpy()
        if self.ctrl_data.DIM == 2:
            mesh_p = meshio.Mesh(
                points=self.x.to_numpy(),
                cells={},
                point_data={
                    "material": self.material.to_numpy().astype(float),
                    "v": self.v.to_numpy(),
                    "Fx": F_np[:, 0, :],
                    "Fy": F_np[:, 1, :],
                    "Sx": Sigma_np[:, 0, :],
                    "Sy": Sigma_np[:, 1, :],
                    "Jp": self.Jp.to_numpy(),
                },
            )
        else:
            # print(self.x.to_numpy().astype(np.float32))
            mesh_p = meshio.Mesh(
                points=self.x.to_numpy().astype(np.float32),
                cells={},
                point_data={
                    "material": self.material.to_numpy().astype(float),
                    "v": self.v.to_numpy(),
                    "Fx": F_np[:, 0, :],
                    "Fy": F_np[:, 1, :],
                    "Fz": F_np[:, 2, :],
                    "Sx": Sigma_np[:, 0, :],
                    "Sy": Sigma_np[:, 1, :],
                    "Sz": Sigma_np[:, 2, :],
                    "S_trace": (Sigma_np[:, 0, 0] + Sigma_np[:, 1, 1] + Sigma_np[:, 2, 2]) / 3.0,
                    "Jp": self.Jp.to_numpy(),
                    "contact_force_normal": self.contact_force_normal_postprocessing.to_numpy().astype(np.float32),
                    "contact_force_tangential": self.contact_force_tangential_postprocessing.to_numpy(),
                },
            )
        dump_path_p = os.path.join(self.ctrl_data.cache_dir, "res", f"particle{frame}.vtk")
        mesh_p.write(dump_path_p)


if __name__ == "__main__":
    pass
