import numpy as np


def np_barycentric_coord(Vs, P):
    # Vs is dim+1,dim mat for dim+1(row) coord of simplex cell
    # P is dim vec for pos
    # return dim+1 vec for barycentric coord
    Vs_rel = Vs[1:] - Vs[0:1]
    P_rel = P - Vs[0]
    cell_coord = Vs_rel.T
    if len(P_rel.shape) == 2:
        # squeeze the dim whose len is 1
        P_rel = P_rel[0]
    coeff = np.linalg.inv(cell_coord).dot(P_rel)
    gamma = 1 - np.sum(coeff)
    BC = np.concatenate((np.array([gamma]), coeff), 0)
    return BC


def flat(candidate_box, min_range, max_range):
    """
    Converts the spatial index represented by `candidate_box` to a scalar int.
    """
    flat_index = 0
    strides = np.cumprod((max_range - min_range)[::-1])[::-1]
    # left shift by 1; then last digit set to 1
    strides[:-1] = strides[1:]
    strides[-1] = 1

    flat_index = np.sum((candidate_box - min_range) * strides, axis=-1)
    return flat_index


def triangle_box_intersect(candidate_box, dx, cell_pos):
    """
    Checks if the candidate box intersects with the 2D triangle defined by `cell_pos`.
    """
    # Convert candidate_box and dx into ndarray representing the nodes position of the box
    box_min = candidate_box * dx
    box_max = box_min + dx

    box_pos = np.array([[box_min[0], box_min[1]], [box_max[0], box_min[1]], [box_max[0], box_max[1]], [box_min[0], box_max[1]]])

    # Call polygon_intersect function
    return polygon_intersect(box_pos, cell_pos)


def tetra_box_intersect(candidate_box, dx, cell_pos):
    """
    Checks if the candidate box intersects with the 3D tetrahedron defined by `cell_pos`.
    """
    # Convert candidate_box and dx into ndarray representing the nodes position of the box
    box_min = candidate_box * dx
    box_max = box_min + dx
    # return True

    # opt 1 there is a node of the tetra in the box
    # opt 2 there is a node of the box in the tetra
    # opt 3 there is a line of the tetra intersecting the box
    # opt 4 there is a line of the box intersecting the tetra
    # if none of the above, then return false
    # opt 1
    eps = 1e-5
    for node in cell_pos:
        if (node >= box_min - eps).all() and (node <= box_max + eps).all():
            return True
    # opt 2
    box_pos = np.array([
        [box_min[0], box_min[1], box_min[2]],
        [box_min[0], box_min[1], box_max[2]],
        [box_min[0], box_max[1], box_min[2]],
        [box_min[0], box_max[1], box_max[2]],
        [box_max[0], box_min[1], box_min[2]],
        [box_max[0], box_min[1], box_max[2]],
        [box_max[0], box_max[1], box_min[2]],
        [box_max[0], box_max[1], box_max[2]],
    ])
    for node in box_pos:
        bcoord = np_barycentric_coord(cell_pos, node)

        if (bcoord >= -eps).all() and (bcoord <= 1 + eps).all():
            return True
    # opt 3
    for i in range(4):
        for j in range(i + 1, 4):
            if line_box_intersect(np.array([box_min, box_max]), cell_pos[i], cell_pos[j]):
                return True
    # opt 4
    box_edges = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]])
    for edge in box_edges:
        e_start = box_pos[edge[0]]
        e_end = box_pos[edge[1]]
        for i in range(4):
            tri_face = np.array([cell_pos[j] for j in range(4) if j != i])
            if line_tri_intersect(tri_face, e_start, e_end):
                return True

    return False


def line_box_intersect(bbox, p1, p2):
    """
    Checks if the line segment defined by `p1` and `p2` intersects with the 3D box defined by `bbox`.
    """
    # for each direction: x y and z
    # check if the line segment intersects with the plane defined by the box
    # if it does, check if the intersection point is inside the box
    # if it is, return true

    for i in range(3):
        if p1[i] == p2[i]:
            continue
        # plane normal
        n = np.zeros(3)
        n[i] = 1
        # line direction
        u = p2 - p1
        # line offset
        w = bbox - p1
        # line parameter
        s = w @ n.reshape((3, 1)) / np.dot(n, u)
        eps = 1e-5
        if (s < -eps).all() or (s > 1 + eps).all():
            continue
        # intersection points on two planes
        p = p1 + s * u
        intp0 = p[0]
        intp1 = p[1]
        # collapse the i direction
        intp0 = np.delete(intp0, i)
        intp1 = np.delete(intp1, i)
        tbbox_min = np.delete(bbox[0], i)
        tbbox_max = np.delete(bbox[1], i)
        if ((intp0 >= tbbox_min - eps).all() and (intp0 <= tbbox_max + eps).all() and s[0] >= -eps and s[0] <= 1 + eps) or ((intp1 >= tbbox_min - eps).all() and
                                                                                                                            (intp1 <= tbbox_max + eps).all() and s[1] >= -eps and s[1] <= 1 + eps):
            return True

    return False


def line_tri_intersect(tri, p1, p2):
    """
    Checks if the line segment defined by `p1` and `p2` intersects with the 3D triangle defined by `tri`.
    """

    # P0 and P1 are the endpoints of the line segment
    # triangle_vertices is a list of three vertices of the triangle

    # calculate the line normal and length
    line_direction = p2 - p1
    line_length = np.linalg.norm(line_direction)
    line_normal = line_direction / line_length

    # Calculate the normal vector of the triangle's plane
    AB = tri[1] - tri[0]
    AC = tri[2] - tri[0]
    triangle_normal = np.cross(AB, AC)
    triangle_normal = triangle_normal / np.linalg.norm(triangle_normal)

    # Check if the line is nearly parallel to the plane
    dot_product = np.dot(line_normal, triangle_normal)
    if np.isclose(dot_product, 0, atol=1e-6):
        return False

    # Calculate the parameter t at which the line intersects the plane
    mat = np.array([tri[0] - tri[2], tri[1] - tri[2], -line_direction]).T
    rel_p = p1 - tri[2]
    a, b, t = np.linalg.inv(mat).dot(rel_p)

    eps = 1e-5
    if a >= -eps and b >= -eps and a + b <= 1 + 2 * eps and t >= -eps and t <= 1 + eps:
        return True
    else:
        return False


def polygon_intersect(poly_pos1, poly_pos2):
    """
    Checks if two polygons defined by `poly_pos1` and `poly_pos2` intersect.
    """
    polygons = [poly_pos1, poly_pos2]
    for i in range(len(polygons)):
        polygon = polygons[i]
        for i1 in range(len(polygon)):
            i2 = (i1 + 1) % len(polygon)
            edge_start = polygon[i1]
            edge_end = polygon[i2]
            edge_normal = (edge_end[1] - edge_start[1], edge_start[0] - edge_end[0])
            minA, maxA = None, None
            for vertex in poly_pos1:
                projected = edge_normal[0] * vertex[0] + edge_normal[1] * vertex[1]
                if minA is None or projected < minA:
                    minA = projected
                if maxA is None or projected > maxA:
                    maxA = projected
            minB, maxB = None, None
            for vertex in poly_pos2:
                projected = edge_normal[0] * vertex[0] + edge_normal[1] * vertex[1]
                if minB is None or projected < minB:
                    minB = projected
                if maxB is None or projected > maxB:
                    maxB = projected
            if maxA < minB or maxB < minA:
                return False
    return True


def intersects(candidate_box, dx, cell_pos):
    """
    Checks if the candidate box intersects with the cell.
    """
    dim = cell_pos.shape[-1]
    if dim == 2:
        return triangle_box_intersect(candidate_box, dx, cell_pos)
    elif dim == 3:
        return tetra_box_intersect(candidate_box, dx, cell_pos)
    else:
        raise ValueError("Unsupported dimension: {}".format(dim))


def spatial_hashmap(pos, cell, dx):
    """
    Generates a nested list of intersecting spatial hash boxes for each cell of the input mesh.
    """
    min_pos = np.min(pos, axis=0)
    max_pos = np.max(pos, axis=0)
    min_range = np.floor(min_pos / dx).astype(int)
    max_range = np.ceil(max_pos / dx).astype(int)
    if (min_range == max_range).all():
        max_range += 1

    hash2cell = [[] for _ in range(np.prod(max_range - min_range))]
    dim = pos.shape[-1]  # Number of dimensions for spatial hash

    if dim == 2:
        for cell_idx in range(len(cell)):
            print("Processing cell {}/{}".format(cell_idx, len(cell)))
            cell_min = np.min(pos[cell[cell_idx]], axis=0)
            cell_max = np.max(pos[cell[cell_idx]], axis=0)
            min_idx = np.floor(cell_min / dx).astype(int)
            max_idx = np.ceil(cell_max / dx).astype(int)

            if (min_idx == max_idx).all():
                max_idx += 1

            candidate_boxes = np.indices(max_idx - min_idx).reshape(dim, -1).T + min_idx

            for candidate_box in candidate_boxes:
                if triangle_box_intersect(candidate_box, dx, pos[cell[cell_idx]]):
                    hash_idx = flat(candidate_box, min_range, max_range)
                    hash2cell[hash_idx].append(cell_idx)

    elif dim == 3:
        for cell_idx in range(len(cell)):
            print("Processing cell {}/{}".format(cell_idx, len(cell)))
            cell_min = np.min(pos[cell[cell_idx]], axis=0)
            cell_max = np.max(pos[cell[cell_idx]], axis=0)
            min_idx = np.floor(cell_min / dx).astype(int)
            max_idx = np.ceil(cell_max / dx).astype(int)

            if (min_idx == max_idx).all():
                max_idx += 1

            candidate_boxes = np.indices(max_idx - min_idx).reshape(dim, -1).T + min_idx
            print('how many candidate boxes: ', candidate_boxes.shape[0])
            hash_idices = flat(candidate_boxes, min_range, max_range)
            for hash_idx in hash_idices:
                hash2cell[hash_idx].append(cell_idx)

    else:
        raise ValueError("Unsupported dimension: {}".format(dim))

    return hash2cell, min_range, max_range


if __name__ == "__main__":
    # test triangle box intersection
    candidate_box = np.array([0., 0.])
    dx = 1.0
    # cell nodes in box
    cell_pos = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5]])
    assert triangle_box_intersect(candidate_box, dx, cell_pos) == True
    # on the box edge
    cell_pos = np.array([[1.0, 0.5], [1.5, 0.5], [1.5, 1.5]])
    assert triangle_box_intersect(candidate_box, dx, cell_pos) == True
    # on the tri edge
    cell_pos = np.array([[0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])
    assert triangle_box_intersect(candidate_box, dx, cell_pos) == True
    # box nodes in tri
    cell_pos = np.array([[0.4, 1.4], [1.4, 0.4], [1.5, 1.5]])
    assert triangle_box_intersect(candidate_box, dx, cell_pos) == True
    # tri in box
    cell_pos = np.array([[0.4, 1.4], [1.4, 0.4], [1.5, 1.5]]) * 0.01
    assert triangle_box_intersect(candidate_box, dx, cell_pos) == True
    # box in tri
    cell_pos = np.array([[-0.5, -0.5], [1.0, 0.], [0., 1.0]])
    assert triangle_box_intersect(candidate_box, dx * 0.2, cell_pos) == True

    # test spatial_hashmap
    pos = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [5.5, 5.5], [6.5, 5.5], [6.5, 6.5]])
    cell = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    dx = 2.0
    cell2hash, hash2cell, min_range, max_range = spatial_hashmap(pos, cell, dx)
    assert (min_range == np.array([0, 0])).all()
    assert (max_range == np.array([4, 4])).all()
    assert cell2hash[0] == [0]
    assert cell2hash[1] == [0]
    assert set(cell2hash[2]) == set([10, 11, 14, 15])
    assert hash2cell[0] == [0, 1]
    assert hash2cell[10] == [2]
    assert hash2cell[11] == [2]
    assert hash2cell[14] == [2]
    assert hash2cell[15] == [2]