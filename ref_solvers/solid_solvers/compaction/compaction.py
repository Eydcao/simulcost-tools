import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Geometry and material
length = 1.0  # m
height = 1.0  # m
nx, ny = 2, 10
E = 1e5        # Young's modulus (Pa)
nu = 0.0       # Poisson's ratio
rho = 1000     # density (kg/m^3)
g = 10         # gravity (m/s^2)
body_force = np.array([0, -rho * g])

hx = length / nx
hy = height / ny
V = hx * hy / 4.0

# Plane strain constitutive matrix
def plane_strain_D(E, nu):
    coeff = E / ((1 + nu) * (1 - 2 * nu))
    return coeff * np.array([
        [1 - nu,     nu,       0],
        [nu,     1 - nu,       0],
        [0,          0, (1 - 2 * nu) / 2]
    ])

D = plane_strain_D(E, nu)

# Mesh generation
def generate_mesh(nx, ny, length, height):
    x = np.linspace(0, length, nx + 1)
    y = np.linspace(0, height, ny + 1)
    xv, yv = np.meshgrid(x, y)
    coords = np.column_stack([xv.flatten(), yv.flatten()])
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = n1 + 1
            n3 = n2 + (nx + 1)
            n4 = n1 + (nx + 1)
            elements.append([n1, n2, n3, n4])
    return np.array(coords), np.array(elements)

coords, elements = generate_mesh(nx, ny, length, height)
n_nodes = coords.shape[0]
n_elements = elements.shape[0]

# Shape functions and derivatives (4-node quad)
def shape_functions(xi, eta):
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

# Gauss points for 2x2 integration
gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)),
                ( 1/np.sqrt(3), -1/np.sqrt(3)),
                ( 1/np.sqrt(3),  1/np.sqrt(3)),
                (-1/np.sqrt(3),  1/np.sqrt(3))]

# Assemble global stiffness matrix and force vector
K = lil_matrix((2 * n_nodes, 2 * n_nodes))
f = np.zeros(2 * n_nodes)

for e in range(n_elements):
    element = elements[e]
    ke = np.zeros((8, 8))
    fe = np.zeros(8)
    xe = coords[element]
    for xi, eta in gauss_points:
        N, dN_dxi = shape_functions(xi, eta)
        J = dN_dxi.T @ xe
        detJ = np.linalg.det(J)
        dN_dx = np.linalg.solve(J, dN_dxi.T).T
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i]     = dN_dx[i, 0]
            B[1, 2*i+1]   = dN_dx[i, 1]
            B[2, 2*i]     = dN_dx[i, 1]
            B[2, 2*i+1]   = dN_dx[i, 0]
        ke += B.T @ D @ B * detJ
        for i in range(4):
            fe[2*i:2*i+2] += N[i] * body_force * detJ
    dofs = np.array([[2*node, 2*node+1] for node in element]).flatten()
    for i in range(8):
        for j in range(8):
            K[dofs[i], dofs[j]] += ke[i, j]
    f[dofs] += fe

# Apply boundary conditions (bottom edge fixed)
tol = 1e-8
fixed_nodes = np.where(coords[:, 1] < tol)[0]
fixed_dofs = np.array([[2*n, 2*n+1] for n in fixed_nodes]).flatten()
free_dofs = np.setdiff1d(np.arange(2 * n_nodes), fixed_dofs)

# Solve the system
K = csr_matrix(K)
u = np.zeros(2 * n_nodes)
u[free_dofs] = spsolve(K[free_dofs][:, free_dofs], f[free_dofs])

# Extract vertical displacement along centerline
centerline_x = length / 2
tolerance = 1e-6
center_nodes = np.where(np.abs(coords[:, 0] - centerline_x) < tolerance)[0]
center_coords = coords[center_nodes]
uy_numerical = u[2 * center_nodes + 1]

# Sort by y-coordinate
sorted_indices = np.argsort(center_coords[:, 1])
y_sorted = center_coords[sorted_indices, 1]
uy_sorted = uy_numerical[sorted_indices]

# Analytical displacement
uy_analytical = -rho * g / E * (1 - 2 * nu**2) * (height * y_sorted - y_sorted**2 / 2)

# Plot comparison
plt.figure()
plt.plot(uy_sorted, y_sorted, 'bo-', label='FEM')
plt.plot(uy_analytical, y_sorted, 'r--', label='Analytical')
plt.xlabel('Vertical displacement (m)')
plt.ylabel('Height (m)')
plt.legend()
plt.title('Vertical displacement along centerline')
plt.grid(True)
plt.tight_layout()
plt.savefig("bar_compaction_displacement_comparison.png", dpi=300)
print("Saved comparison figure as 'bar_compaction_displacement_comparison.png'.")

# # Optional: print comparison table
# for yi, ui_fem, ui_ana in zip(y_sorted, uy_sorted, uy_analytical):
#     print(f"y = {yi:.3f} m | FEM uy = {ui_fem:.6f} m | Analytical uy = {ui_ana:.6f} m")


# --- Stress comparison at Gauss points ---

gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)),
                ( 1/np.sqrt(3), -1/np.sqrt(3)),
                ( 1/np.sqrt(3),  1/np.sqrt(3)),
                (-1/np.sqrt(3),  1/np.sqrt(3))]

stress_data = []

for e in range(n_elements):
    element = elements[e]
    xe = coords[element]
    ue = ue = np.array([u[2*n + i] for n in element for i in range(2)]) #ue = np.array([u[2*n], u[2*n+1]] for n in element).flatten()
    # if e==0:
    #     print(ue)
    for xi, eta in gauss_points:
        _, dN_dxi = shape_functions(xi, eta)
        J = dN_dxi.T @ xe
        dN_dx = np.linalg.solve(J, dN_dxi.T).T
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i]     = dN_dx[i, 0]
            B[1, 2*i+1]   = dN_dx[i, 1]
            B[2, 2*i]     = dN_dx[i, 1]
            B[2, 2*i+1]   = dN_dx[i, 0]
        # if e==0:
        #     print(xi, eta, B)
        strain = B @ ue
        stress = D @ strain
        sigma_yy = stress[1]
        # Map Gauss point to physical coordinates
        N, _ = shape_functions(xi, eta)
        x_gp = N @ xe[:, 0]
        y_gp = N @ xe[:, 1]
        sigma_yy_analytical = rho * g * (y_gp - height)
        stress_data.append([y_gp, sigma_yy, sigma_yy_analytical])

# Sort and print comparison table
stress_data = np.array(stress_data)
stress_data = stress_data[stress_data[:, 0].argsort()]  # sort by y

# print("y (m)    |  FEM σ_yy (Pa)  |  Analytical σ_yy (Pa)")
# print("-----------------------------------------------")
# for row in stress_data:
#     print(f"{row[0]:.4f}  |  {row[1]:+.6e}  |  {row[2]:+.6e}")



# --- Plot vertical stress comparison ---
import matplotlib.pyplot as plt

# Sort by vertical position
stress_data = np.array(stress_data)
stress_data = stress_data[stress_data[:, 0].argsort()]  # sort by y

# Extract values
y_vals = stress_data[:, 0]
sigma_fem = stress_data[:, 1]
sigma_ana = stress_data[:, 2]

# Plot
plt.figure(figsize=(6, 5))
plt.plot(sigma_fem, y_vals, 'bo-', label='FEM σ_yy')
plt.plot(sigma_ana, y_vals, 'r--', label='Analytical σ_yy')
plt.xlabel('Vertical Stress σ_yy (Pa)')
plt.ylabel('Height y (m)')
plt.title('Vertical Stress Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vertical_stress_comparison.png", dpi=300)
plt.show()

# Calculate error
error_numerator = np.sqrt(np.sum((sigma_fem - sigma_ana)**2)) * V
error_denominator = g * rho * height * V * len(y_vals)
print('Error:', error_numerator/error_denominator)