import taichi as ti
import numpy as np
from .hash_helpers import spatial_hashmap
from enum import Enum
import time


class ADV_TYPE(Enum):
    FLIP = 0
    PIC = 1
    APIC = 2


ADV_TYPE_STR = ["flip", "pic", "apic"]


@ti.func
def B_spline(d: float):
    # B spline kernel
    coeff = 0.0
    if d <= 0.5 and d >= -0.5:
        coeff = 0.75 - d**2
    elif d < -0.5 and d >= -1.5:
        coeff = 0.5 * (d + 1.5) ** 2
    elif d > 0.5 and d <= 1.5:
        coeff = 0.5 * (d - 1.5) ** 2

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
    print("v_adj_f time", time.time() - start_time)

    # bc node and normal detection
    start_time = time.time()
    is_bc = [False for i in range(v_pos.shape[0])]
    outgoing_normal = [np.zeros(DIM) for i in range(v_pos.shape[0])]
    for i in range(v_pos.shape[0]):
        is_bc[i], outgoing_normal[i] = detect_if_bc_node(i, cell, v_adj_f, v_pos)

    # exit(1)
    # convert to np
    is_bc = np.array(is_bc, dtype=int)
    outgoing_normal = np.array(outgoing_normal)
    print("bc node detection time", time.time() - start_time)

    # convert to list
    start_time = time.time()
    v_adj_f = [list(adj) for adj in v_adj_f]
    print("v_adj_f time convert to list", time.time() - start_time)

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
    print("f_1b_v time", time.time() - start_time)

    # convert to csr format
    start_time = time.time()
    f_1b_v_idx, f_1b_v_var, max_nonzero_f_1b_v = convert_to_csr(f_1b_v)
    print("f_1b_v time convert to csr", time.time() - start_time)

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
    print("v_adj_v time", time.time() - start_time)

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
    print("f_1b_v_connectivity_2_f_v time", time.time() - start_time)

    # construct hash map
    # hash_vec = np.array([hash_dx, hash_dy])
    start_time = time.time()
    hash2cell, hash_min, hash_max = spatial_hashmap(v_pos / hash_vec, cell, 1.0)
    print("hash time", time.time() - start_time)
    # hash2cell is again a nested list containing the cell indices that intersect with this hash box
    # convert it to csr format just like v_adj_cell_idx/var
    start_time = time.time()
    hash2cell_idx, hash2cell_var, _ = convert_to_csr(hash2cell)
    print("hash time convert to csr", time.time() - start_time)

    # create sizing field
    start_time = time.time()
    # for each cell, get all the N0 nodes, and all the N1 nodes
    # calcualte the maximum distance between all possible pairs between N0 and N1
    v_support_radii = np.zeros(v_pos.shape[0])
    for i, c in enumerate(cell):
        N0 = v_pos[c]
        N1 = v_pos[np.array(f_1b_v[i])]
        diff = N0[:, None, :] - N1[None, :, :]
        diff_d = np.linalg.norm(diff, axis=-1)
        max_d = np.max(diff_d)
        # for each vertex of c, assign max(max_d, v_support_radii) to v_support_radii
        for v_idx in c:
            v_support_radii[v_idx] = max(max_d, v_support_radii[v_idx])

    # print(v_support_radii)
    # from matplotlib import pyplot as plt
    # import meshio
    # assign the support radii to the mesh and save a temp.vtk for visualization
    # create an empty mesh
    # mesh = meshio.Mesh(points=v_pos, cells={'triangle': cell}, point_data={'support_radii': v_support_radii})
    # write the mesh to file
    # meshio.write('./temp.vtk', mesh)
    # plt.plot(v_support_radii)
    # plt.savefig('./v_support_radii.png')
    # exit(1)

    return (
        is_bc,
        outgoing_normal,
        f_1b_v_idx,
        f_1b_v_var,
        max_nonzero_f_1b_v,
        f_1b_v_connectivity_2_f_v.astype(float),
        hash2cell_idx,
        hash2cell_var,
        hash_min,
        hash_max,
        v_support_radii,
    )


def convert_to_csr(nested_list):
    # convert a nested list into csr format
    # return the index and var array, and maximum nonzero elements in a row
    lens = np.array([len(adj) for adj in nested_list])
    idx = np.cumsum(np.insert(lens, 0, 0))
    var = np.array([v_idx for adj in nested_list for v_idx in adj])

    return idx, var, np.max(lens)


@ti.func
def barycentric_coord_1d(T0, T1, P):
    len = (T1 - T0).norm()
    coeff = (P - T0).norm() / len
    BC = ti.Vector([1.0 - coeff, coeff])
    return BC


@ti.func
def barycentric_coord_2d(T0, T1, T2, P):
    # determinant of original triangle
    Mat = ti.Matrix([[1.0, T0[0], T0[1]], [1.0, T1[0], T1[1]], [1.0, T2[0], T2[1]]]).cast(ti.f32)
    D = Mat.determinant()
    # determinant of using P to replace each vertex
    D0 = ti.Matrix([[1.0, P[0], P[1]], [1.0, T1[0], T1[1]], [1.0, T2[0], T2[1]]]).cast(ti.f32).determinant()
    D1 = ti.Matrix([[1.0, T0[0], T0[1]], [1.0, P[0], P[1]], [1.0, T2[0], T2[1]]]).cast(ti.f32).determinant()
    D2 = D - D0 - D1
    BC = ti.Vector([D0, D1, D2]) / D
    return BC


@ti.func
def barycentric_coord_3d(v0, v1, v2, v3, P):
    # determinant of original tetra
    Mat = ti.Matrix(
        [
            [1.0, v0[0], v0[1], v0[2]],
            [1.0, v1[0], v1[1], v1[2]],
            [1.0, v2[0], v2[1], v2[2]],
            [1.0, v3[0], v3[1], v3[2]],
        ]
    ).cast(ti.f32)
    D = Mat.determinant()
    # determinant of using P to replace each vertex
    D0 = (
        ti.Matrix(
            [
                [1.0, P[0], P[1], P[2]],
                [1.0, v1[0], v1[1], v1[2]],
                [1.0, v2[0], v2[1], v2[2]],
                [1.0, v3[0], v3[1], v3[2]],
            ]
        )
        .cast(ti.f64)
        .determinant()
    )
    D1 = (
        ti.Matrix(
            [
                [1.0, v0[0], v0[1], v0[2]],
                [1.0, P[0], P[1], P[2]],
                [1.0, v2[0], v2[1], v2[2]],
                [1.0, v3[0], v3[1], v3[2]],
            ]
        )
        .cast(ti.f64)
        .determinant()
    )
    D2 = (
        ti.Matrix(
            [
                [1.0, v0[0], v0[1], v0[2]],
                [1.0, v1[0], v1[1], v1[2]],
                [1.0, P[0], P[1], P[2]],
                [1.0, v3[0], v3[1], v3[2]],
            ]
        )
        .cast(ti.f64)
        .determinant()
    )
    D3 = D - D0 - D1 - D2
    BC = ti.Vector([D0, D1, D2, D3]) / D
    return BC


def mesh2line(v_pos_np, cell_np):
    # get flat edge positions (for visualization only)
    T_pos_np = v_pos_np[cell_np, :]
    T_pos_edge_line = np.concatenate(
        (
            T_pos_np[:, 0:2],
            T_pos_np[:, 1:3],
            np.concatenate((T_pos_np[:, 2:3], T_pos_np[:, 0:1]), 1),
        ),
        0,
    )

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
        outgoing_normal: Array of outgoing normals for the boundary edges
    """
    DIM = pos.shape[-1]
    # bc_ means the boundary for each cell, in 2d this is an edge, in 3d this is a face
    bc_cell_adj_cell = {}

    if DIM == 2:
        for c in v_adj_c[p_idx]:
            for adj_n_idx in cell[c]:
                if adj_n_idx != p_idx:
                    edge_key = tuple(sorted((p_idx, adj_n_idx)))
                    if edge_key not in bc_cell_adj_cell:
                        bc_cell_adj_cell[edge_key] = [c]
                    else:
                        bc_cell_adj_cell[edge_key].append(c)

        is_bc = False
        outgoing_normal = np.zeros(DIM)
        sumed_area = 0.0

        for key, value in bc_cell_adj_cell.items():
            value = list(set(value))  # Remove duplicate adjacent cell indices
            if len(value) == 1:
                is_bc = True
                temp_adj_node_idx = key[0] if key[1] == p_idx else key[1]
                edge_vector = pos[p_idx] - pos[temp_adj_node_idx]
                p_to_cell_enter = pos[cell[value[0]]].mean(axis=0) - pos[p_idx]
                aread_normal = np.array(
                    [-edge_vector[1], edge_vector[0]]
                )  # Calculate the normal orthogonal to the edge
                if np.dot(aread_normal, p_to_cell_enter) > 0:
                    aread_normal *= -1  # Flip the normal if it points inside
                outgoing_normal += aread_normal
                sumed_area += np.linalg.norm(aread_normal)

        if is_bc:
            outgoing_normal /= np.linalg.norm(outgoing_normal)

    elif DIM == 3:
        for c in v_adj_c[p_idx]:
            # choose 2 out of 3 vertices of this tetra (not including p_idx);
            # use these 2 vertices and p_idx to form a face
            for i in range(4):
                if cell[c, i] != p_idx:
                    for j in range(i + 1, 4):
                        if cell[c, j] != p_idx and cell[c, i] != cell[c, j]:
                            face_key = tuple(sorted((p_idx, cell[c, i], cell[c, j])))
                            if face_key not in bc_cell_adj_cell:
                                bc_cell_adj_cell[face_key] = set([c])
                            else:
                                bc_cell_adj_cell[face_key].add(c)

        is_bc = False
        outgoing_normal = np.zeros(DIM)
        sumed_area = 0.0

        added_normal = []
        added_tet_center = []

        for key, value in bc_cell_adj_cell.items():
            value = list(value)  # Remove duplicate adjacent cell indices
            if len(value) == 1:
                is_bc = True
                # # get the center of the face
                # face_center = pos[list(key)].mean(axis=0)
                # get the center of the cell
                cell_center = pos[cell[value[0]]].mean(axis=0)
                f2c = cell_center - pos[p_idx]  # used to flip the normal if necessary
                # get the normal of the face
                aread_normal = np.cross(pos[key[1]] - pos[key[0]], pos[key[2]] - pos[key[0]])
                if np.dot(aread_normal, f2c) > 0:
                    aread_normal *= -1  # Flip the normal if it points inside

                added_normal.append(aread_normal)
                # added_tet_center.append(pos[cell[value[0]]])
                added_tet_center.append(cell[value[0]])

                outgoing_normal += aread_normal
                sumed_area += np.linalg.norm(aread_normal)

        if is_bc:
            # if np.linalg.norm(outgoing_normal) < 1e-4:
            #     # print('outgoing_normal, area', outgoing_normal, sumed_area)
            #     print('v ipx', p_idx)
            #     print('v pos', pos[p_idx])
            #     print('map', bc_cell_adj_cell)
            #     print('added_normal', added_normal)
            #     print('added_tet_center', added_tet_center)
            #     for i in added_tet_center[0]:
            #         print(pos[i])
            #     # exit(1)

            outgoing_normal /= np.linalg.norm(outgoing_normal)

    else:
        raise NotImplementedError

    return is_bc, outgoing_normal


@ti.kernel
def test_barycentric_coord_2d():
    T0 = ti.Vector([0.0, 0.0])
    T1 = ti.Vector([1.0, 0.0])
    T2 = ti.Vector([0.0, 1.0])
    P = ti.Vector([0.5, 0.5])
    print(barycentric_coord_2d(T0, T1, T2, P))

    P = ti.Vector([0.0, 0.0])
    print(barycentric_coord_2d(T0, T1, T2, P))

    P = ti.Vector([1.0, 0.0])
    print(barycentric_coord_2d(T0, T1, T2, P))

    P = ti.Vector([0.0, 1.0])
    print(barycentric_coord_2d(T0, T1, T2, P))

    P = ti.Vector([1.0, 1.0])
    print(barycentric_coord_2d(T0, T1, T2, P))

    P = ti.Vector([-1.0, -1.0])
    print(barycentric_coord_2d(T0, T1, T2, P))

    P = ti.Vector([1.0, -1.0])
    print(barycentric_coord_2d(T0, T1, T2, P))

    P = ti.Vector([-1.0, 1.0])
    print(barycentric_coord_2d(T0, T1, T2, P))

    P = ti.Vector([1.5, -0.5])
    print(barycentric_coord_2d(T0, T1, T2, P))

    P = ti.Vector([-0.5, 1.5])
    print(barycentric_coord_2d(T0, T1, T2, P))


@ti.kernel
def test_barycentric_coord_3d():
    T0 = ti.Vector([0.0, 0.0, 0.0])
    T1 = ti.Vector([1.0, 0.0, 0.0])
    T2 = ti.Vector([0.0, 1.0, 0.0])
    T3 = ti.Vector([0.0, 0.0, 1.0])

    P = ti.Vector([0.5, 0.5, 0.5])
    print(barycentric_coord_3d(T0, T1, T2, T3, P))

    P = ti.Vector([0.25, 0.25, 0.25])
    print(barycentric_coord_3d(T0, T1, T2, T3, P))

    P = ti.Vector([0.25, 0.25, 0.25]) * -1
    print(barycentric_coord_3d(T0, T1, T2, T3, P))

    P = ti.Vector([-0.25, 0.25, 0.25])
    print(barycentric_coord_3d(T0, T1, T2, T3, P))

    P = ti.Vector([0.25, -0.25, 0.25])
    print(barycentric_coord_3d(T0, T1, T2, T3, P))

    P = ti.Vector([0.25, 0.25, -0.25])
    print(barycentric_coord_3d(T0, T1, T2, T3, P))

    P = ti.Vector([-0.25, 0.25, 0.25]) * -1
    print(barycentric_coord_3d(T0, T1, T2, T3, P))

    P = ti.Vector([0.25, -0.25, 0.25]) * -1
    print(barycentric_coord_3d(T0, T1, T2, T3, P))

    P = ti.Vector([0.25, 0.25, -0.25]) * -1
    print(barycentric_coord_3d(T0, T1, T2, T3, P))


if __name__ == "__main__":
    import taichi as ti
    import numpy as np

    # test barycentric_coord_2d and barycentric_coord_3d
    ti.init(arch=ti.cpu)

    # test_barycentric_coord_2d()
    test_barycentric_coord_3d()
