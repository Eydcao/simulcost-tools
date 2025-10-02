import taichi as ti
import numpy as np
from hash_helpers import spatial_hashmap
from enum import Enum
import time


class ADV_TYPE(Enum):
    FLIP = 0
    PIC = 1
    APIC = 2


ADV_TYPE_STR = ['flip', 'pic', 'apic']


@ti.func
def B_spline(d: float):
    # B spline kernel
    coeff = 0.0
    if d < 0.5 and d > -0.5:
        coeff = 0.75 - d**2
    elif d < -0.5 and d > -1.5:
        coeff = 0.5 * (d + 1.5)**2
    elif d > 0.5 and d < 1.5:
        coeff = 0.5 * (d - 1.5)**2

    return coeff


@ti.func
def pow_sep(d: float):
    coeff = ti.exp(d**2) - 1.0

    return coeff


@ti.func
def sqr_sep(d: float):
    coeff = d**2

    return coeff


def build_connectivity(v_pos, cell, hash_vec):
    start_time = time.time()

    # get DIM
    DIM = v_pos.shape[-1]
    # init v-f connectivity
    v_adj_f = [set() for i in range(v_pos.shape[0])]
    for f_idx in range(cell.shape[0]):
        for v_idx in cell[f_idx]:
            v_adj_f[v_idx].add(f_idx)
    print('v_adj_f time', time.time() - start_time)

    # convert to list
    start_time = time.time()
    v_adj_f = [list(adj) for adj in v_adj_f]
    print('v_adj_f time convert to list', time.time() - start_time)

    # init f-1b-v/f connectivity, f-1b-f is the unioined set of all adj fs around this f; f-1b-v is the unioned set of all the v_adj_f's vertices of the f's vertices
    start_time = time.time()
    f_1b_v = [set() for i in range(cell.shape[0])]
    for f_idx in range(cell.shape[0]):
        # 1st union all adj fs
        all_adj_f = set()
        for v_idx in cell[f_idx]:
            all_adj_f = all_adj_f.union(set(v_adj_f[v_idx]))
        # then use all_adj_f and cell to union all the vertices
        for adj_f_idx in all_adj_f:
            for v_idx in cell[adj_f_idx]:
                f_1b_v[f_idx].add(v_idx)
        # set the f_1b_f
    # convert to list
    f_1b_v = [list(adj) for adj in f_1b_v]
    print('f_1b_v time', time.time() - start_time)

    # convert to csr format
    start_time = time.time()
    f_1b_v_idx, f_1b_v_var, max_nonzero_f_1b_v = convert_to_csr(f_1b_v)
    print('f_1b_v time convert to csr', time.time() - start_time)

    # create v-v connectivity for later use
    start_time = time.time()
    v_adj_v = [set() for i in range(v_pos.shape[0])]
    # for each f of c
    # convert all this f's v to a set
    # for each v_idx in this set, union v_adj_v[v_idx] with this set
    for f_idx in range(cell.shape[0]):
        tmp_cell = cell[f_idx]
        for v_idx in tmp_cell:
            v_adj_v[v_idx] = v_adj_v[v_idx].union(set(tmp_cell))
    print('v_adj_v time', time.time() - start_time)

    # define a field with size of (f_1b_f_var, DIM+1)
    # each entry (row=a certain f's a certain adj 1b v) represents if the v is connected to the cell f's ith v (i=col=0,...DIM+1)
    start_time = time.time()
    f_1b_v_connectivity_2_f_v = np.zeros((f_1b_v_var.shape[0], DIM + 1), dtype=bool)
    for f_idx in range(cell.shape[0]):
        # get the cell
        tmp_cell = cell[f_idx]
        # get the csr index of v
        for adj_v_idx_csr in range(f_1b_v_idx[f_idx], f_1b_v_idx[f_idx + 1]):
            # get v by f_1b_v_var
            tmp_v = f_1b_v_var[adj_v_idx_csr]
            # for each vertex of the cell, v_cell, if tmp_v is in v_adj_v[v_cell]
            # then this entry is true
            for i in range(DIM + 1):
                f_1b_v_connectivity_2_f_v[adj_v_idx_csr, i] = tmp_v in v_adj_v[tmp_cell[i]]
    print('f_1b_v_connectivity_2_f_v time', time.time() - start_time)

    # construct hash map
    # hash_vec = np.array([hash_dx, hash_dy])
    start_time = time.time()
    hash2cell, hash_min, hash_max = spatial_hashmap(v_pos / hash_vec, cell, 1.0)
    print('hash time', time.time() - start_time)
    # hash2cell is again a nested list containing the cell indices that intersect with this hash box
    # convert it to csr format just like v_adj_cell_idx/var
    start_time = time.time()
    hash2cell_idx, hash2cell_var, _ = convert_to_csr(hash2cell)
    print('hash time convert to csr', time.time() - start_time)

    return f_1b_v_idx, f_1b_v_var, max_nonzero_f_1b_v, f_1b_v_connectivity_2_f_v, hash2cell_idx, hash2cell_var, hash_min, hash_max


def convert_to_csr(nested_list):
    # convert a nested list into csr format
    # return the index and var array, and maximum nonzero elements in a row
    lens = np.array([len(adj) for adj in nested_list])
    idx = np.cumsum(np.insert(lens, 0, 0))
    var = np.array([v_idx for adj in nested_list for v_idx in adj])

    return idx, var, np.max(lens)


@ti.func
def barycentric_coord_2d(T0, T1, T2, P):
    # T 3,2 mat for 3(row) coord of triangle
    # P 2 vec for pos
    CA = T0 - T2
    CB = T1 - T2
    CP = P - T2
    # cat edge vec in col
    Tri_cood = ti.math.mat2(CA, CB).transpose()
    coeff = Tri_cood.inverse() @ CP
    gamma = 1 - coeff[0] - coeff[1]
    BC = ti.Vector([coeff[0], coeff[1], gamma])
    return BC


@ti.func
def barycentric_coord_3d(v0, v1, v2, v3, P):
    # T 3,2 mat for 3(row) coord of triangle
    # P 2 vec for pos
    v01 = v1 - v0
    v02 = v2 - v0
    v03 = v3 - v0
    v0p = P - v0
    # cat edge vec in col
    cell_cood = ti.math.mat3(v01, v02, v03).transpose()
    coeff = cell_cood.inverse() @ v0p
    gamma = 1 - coeff[0] - coeff[1] - coeff[2]
    BC = ti.Vector([gamma, coeff[0], coeff[1], coeff[2]])
    return BC


def mesh2line(v_pos_np, cell_np):
    # get flat edge positions (for visualization only)
    T_pos_np = v_pos_np[cell_np, :]
    T_pos_edge_line = np.concatenate((T_pos_np[:, 0:2], T_pos_np[:, 1:3], np.concatenate((T_pos_np[:, 2:3], T_pos_np[:, 0:1]), 1)), 0)

    return T_pos_edge_line


def detect_if_bc_node(p_idx, cell, v_adj_c, pos):
    """
    Detects if a node is a boundary node and returns relevant information.

    Args:
        p_idx: Index of the current node
        cell: Connectivity array of the mesh cells
        v_adj_c: Nested list of adjacent cell indices for each vertex
        pos: Array of mesh vertex positions

    Returns:
        is_bc: Boolean indicating if the node is a boundary node
        adj_bc_cell_idx: Array of adjacent cell indices with the boundary edges
        adj_bc_node_idx: Array of adjacent node indices forming the boundary edges
        outgoing_normals: Array of outgoing normals for the boundary edges
    """
    edge_adj_cell = {}

    for c in v_adj_c[p_idx]:
        for adj_n_idx in cell[c]:
            if adj_n_idx != p_idx:
                edge_key = tuple(sorted((p_idx, adj_n_idx)))
                if edge_key not in edge_adj_cell:
                    edge_adj_cell[edge_key] = [c]
                else:
                    edge_adj_cell[edge_key].append(c)

    adj_bc_cell_idx = []
    adj_bc_node_idx = []
    outgoing_normals = []

    for key, value in edge_adj_cell.items():
        value = list(set(value))  # Remove duplicate adjacent cell indices
        if len(value) == 1:
            temp_bc_node_idx = key[0] if key[1] == p_idx else key[1]
            adj_bc_cell_idx.append(value[0])
            adj_bc_node_idx.append(temp_bc_node_idx)
            # NOTE this is for 2D only
            # TODO extend it to 3D
            edge_vector = pos[p_idx] - pos[temp_bc_node_idx]
            p_to_cell_enter = pos[cell[value[0]]].mean(axis=0) - pos[p_idx]
            normal = np.array([-edge_vector[1], edge_vector[0]])  # Calculate the normal orthogonal to the edge
            normal /= np.sqrt(normal[0]**2 + normal[1]**2)  # Normalize the normal vector
            if np.dot(normal, p_to_cell_enter) > 0:
                normal *= -1  # Flip the normal if it points inside
            outgoing_normals.append(normal)

    is_bc = len(adj_bc_cell_idx) > 0

    return is_bc, np.array(adj_bc_cell_idx), np.array(adj_bc_node_idx), np.array(outgoing_normals)


if __name__ == "__main__":
    pass