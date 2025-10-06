import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Parameters
hx = 0.05 # m
hy = 0.05 # m
nx, ny = 40, 40 #160,160 #120, 120  # mesh resolution
length = hx * nx # half width (quarter model)
height = hy * ny # half height

R = 0.5       # hole radius, m

V = hx * hy / 4.0
E = 1e5 # MPa
nu = 0.3
traction = 1000  # MPa

# Material matrix for plane strain
def plane_strain_D(E, nu):
    coeff = E / ((1 + nu) * (1 - 2 * nu))
    return coeff * np.array([
        [1 - nu,     nu,       0],
        [nu,     1 - nu,       0],
        [0,          0, (1 - 2 * nu) / 2]
    ])
D = plane_strain_D(E, nu)

# Generate structured mesh and remove inner hole
x = np.linspace(0, length, nx + 1)
y = np.linspace(0, height, ny + 1)
xv, yv = np.meshgrid(x, y)
coords = np.column_stack([xv.flatten(), yv.flatten()])
elements = []

node_id = lambda i, j: j * (nx + 1) + i

n_nodes = coords.shape[0]
activated_nodes = np.zeros(n_nodes) # Activated nodes

for j in range(ny):
    for i in range(nx):
        n1 = node_id(i, j)
        n2 = node_id(i + 1, j)
        n3 = node_id(i + 1, j + 1)
        n4 = node_id(i, j + 1)
        xc = np.mean([coords[n1, 0], coords[n2, 0], coords[n3, 0], coords[n4, 0]])
        yc = np.mean([coords[n1, 1], coords[n2, 1], coords[n3, 1], coords[n4, 1]])
        if np.sqrt(xc**2 + yc**2) >= R:
            elements.append([n1, n2, n3, n4])

            # Setup for activated nodes
            activated_nodes[n1] = 1.0
            activated_nodes[n2] = 1.0
            activated_nodes[n3] = 1.0
            activated_nodes[n4] = 1.0

elements = np.array(elements)



# Shape functions and derivatives
def shape_functions(xi, eta):
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

# Gauss points
gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)),
                ( 1/np.sqrt(3), -1/np.sqrt(3)),
                ( 1/np.sqrt(3),  1/np.sqrt(3)),
                (-1/np.sqrt(3),  1/np.sqrt(3))]

# Assembly
K = lil_matrix((2 * n_nodes, 2 * n_nodes))
f = np.zeros(2 * n_nodes)

for e in range(len(elements)):
    element = elements[e]
    xe = coords[element]
    ke = np.zeros((8, 8))
    fe = np.zeros(8)
    for xi, eta in gauss_points:
        N, dN_dxi = shape_functions(xi, eta)
        J = xe.T @ dN_dxi
        detJ = np.linalg.det(J)
        dN_dx = np.linalg.solve(J, dN_dxi.T).T
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i]     = dN_dx[i, 0]
            B[1, 2*i+1]   = dN_dx[i, 1]
            B[2, 2*i]     = dN_dx[i, 1]
            B[2, 2*i+1]   = dN_dx[i, 0]
        ke += B.T @ D @ B * detJ
    dofs = np.array([[2*node, 2*node+1] for node in element]).flatten()
    for i in range(8):
        for j in range(8):
            K[dofs[i], dofs[j]] += ke[i, j]

# Apply Neumann BCs (traction on right and top)
tol = 1e-8
def apply_edge_traction(edge_nodes, traction_vec):
    for i in range(len(edge_nodes)-1):
        n1, n2 = edge_nodes[i], edge_nodes[i+1]
        x1, y1 = coords[n1]
        x2, y2 = coords[n2]
        length_edge = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        fe = np.zeros(4)
        fe[0:2] = traction_vec * length_edge / 2
        fe[2:4] = traction_vec * length_edge / 2
        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        for j in range(4):
            f[dofs[j]] += fe[j]

# Find edges
right_nodes = np.where(np.abs(coords[:, 0] - length) < tol)[0]
top_nodes   = np.where(np.abs(coords[:, 1] - height) < tol)[0]
right_nodes = sorted(right_nodes, key=lambda i: coords[i,1])
top_nodes   = sorted(top_nodes, key=lambda i: coords[i,0])

apply_edge_traction(right_nodes, np.array([traction, 0]))
# apply_edge_traction(top_nodes,   np.array([0, traction]))

# Apply Dirichlet BCs (symmetry)
bc_dofs = []
for i in range(n_nodes):
    x, y = coords[i]
    if np.abs(x) < tol:
        bc_dofs.append(2*i)     # ux = 0
    if np.abs(y) < tol:
        bc_dofs.append(2*i + 1) # uy = 0

    # Set the deactivated nodes as boundary nodes
    if activated_nodes[i] < 0.5:
        bc_dofs.append(2*i)
        bc_dofs.append(2*i + 1)

bc_dofs = np.array(bc_dofs)
free_dofs = np.setdiff1d(np.arange(2 * n_nodes), bc_dofs)

# Solve
K = csr_matrix(K)
u = np.zeros(2 * n_nodes)
u[free_dofs] = spsolve(K[free_dofs][:, free_dofs], f[free_dofs])

# Postprocess - plot deformed shape
scale = 10  # exaggerate deformation
x_def = coords[:, 0] + scale * u[0::2]
y_def = coords[:, 1] + scale * u[1::2]

plt.figure(figsize=(6,6))
for e in elements:
    x0 = x_def[e]
    y0 = y_def[e]
    plt.fill(x0, y0, edgecolor='k', fill=False, linewidth=0.5)
plt.axis('equal')
plt.title("Deformed shape (scale ×{})".format(scale))
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()
plt.savefig("deformed_plate_with_hole.png", dpi=300)
plt.show()



# --- Stress extraction along x = 0 ---
sigma_data = []
for e in elements:
    xe = coords[e]
    ue = np.array([u[2*n + i] for n in e for i in range(2)])
    for xi, eta in gauss_points:
        N, dN_dxi = shape_functions(xi, eta)
        J = xe.T @ dN_dxi
        dN_dx = np.linalg.solve(J, dN_dxi.T).T
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i]     = dN_dx[i, 0]
            B[1, 2*i+1]   = dN_dx[i, 1]
            B[2, 2*i]     = dN_dx[i, 1]
            B[2, 2*i+1]   = dN_dx[i, 0]
        strain = B @ ue
        stress = D @ strain
        sigma_xx = stress[0]
        x_gp = N @ xe[:, 0]
        y_gp = N @ xe[:, 1]
        if abs(x_gp) < hx/2 and y_gp > R + 1e-6:
            sigma_xx_ana = traction * (1 + R**2/2/y_gp**2 + 3*R**4/2/y_gp**4) #traction * (1 - R**2 / y_gp**2)
            sigma_data.append([y_gp, sigma_xx, sigma_xx_ana])

# Sort by y
sigma_data = np.array(sigma_data)
sigma_data = sigma_data[sigma_data[:, 0].argsort()]
y_vals = sigma_data[:, 0]
sigma_fem = sigma_data[:, 1]
sigma_ana = sigma_data[:, 2]

# Plot
plt.figure(figsize=(6,5))
plt.plot(y_vals, sigma_fem, 'bo-', label='FEM σ_xx')
plt.plot(y_vals, sigma_ana, 'r--', label='Analytical σ_xx')
plt.xlabel('y (m)')
plt.ylabel('σ_xx (MPa)')
plt.title('Horizontal Stress σ_xx along x = 0')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sigma_xx_comparison.png", dpi=300)
plt.show()


# Error
error_numerator = np.sqrt(np.sum((sigma_fem-sigma_ana)**2)) * V
error_denominator = traction * V * len(sigma_fem)
print('Error:', error_numerator/error_denominator)


# --- Evaluate sigma_xx at element centroids ---
centroids = []
sigma_xx_vals = []

for e in elements:
    xe = coords[e]
    ue = np.array([u[2*n + i] for n in e for i in range(2)])

    # Use center of element for plotting
    xi, eta = 0.0, 0.0  # center of reference element
    N, dN_dxi = shape_functions(xi, eta)
    J = xe.T @ dN_dxi
    dN_dx = np.linalg.solve(J, dN_dxi.T).T
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2*i]     = dN_dx[i, 0]
        B[1, 2*i+1]   = dN_dx[i, 1]
        B[2, 2*i]     = dN_dx[i, 1]
        B[2, 2*i+1]   = dN_dx[i, 0]
    strain = B @ ue
    stress = D @ strain
    sigma_xx = stress[0]

    x_cent = np.mean(xe[:, 0])
    y_cent = np.mean(xe[:, 1])

    if np.sqrt(x_cent**2 + y_cent**2) >= R:
        centroids.append([x_cent, y_cent])
        sigma_xx_vals.append(sigma_xx)

centroids = np.array(centroids)
sigma_xx_vals = np.array(sigma_xx_vals)

# --- Contour plot ---
fig, ax = plt.subplots(figsize=(6, 5))
for e_idx, e in enumerate(elements):
    xe = coords[e]
    stress = sigma_xx_vals[e_idx]
    ax.fill(xe[:, 0], xe[:, 1], facecolor=plt.cm.jet((stress - sigma_xx_vals.min()) / (sigma_xx_vals.max() - sigma_xx_vals.min())), edgecolor='k', linewidth=0.1)

sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=3200))
sm.set_array([])  # Needed for ScalarMappable to work in colorbar
fig.colorbar(sm, ax=ax, label=r'$\sigma_{xx}$ (Pa)')

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(r"Horizontal Stress $\sigma_{xx}$ (MPa) per Element")
ax.axis('equal')
plt.tight_layout()
plt.savefig("sigma_xx_fill_elements.png", dpi=300)
plt.show()