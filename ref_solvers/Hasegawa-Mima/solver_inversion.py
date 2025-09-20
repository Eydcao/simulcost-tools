import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, linalg

# Parameters
L = 2*np.pi*10
N = 256
dx = L / N
dy = L / N
dt = 5e1
tmax = 5e2
v_star = 2e-2

x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Initial condition
phi = np.exp(-((X - L/2)**2 + (Y - L/2)**2)/2**2)

# Initial condition: symmetric dipole with weak overlap
# phi = (
#     np.exp(-((X - (L/2 - 0.12))**2 + (Y - L/2)**2) / 0.09**2)
#     - np.exp(-((X - (L/2 + 0.12))**2 + (Y - L/2)**2) / 0.09**2)
# )

# Laplacian operator in 1D with periodic BC
e = np.ones(N)
L1D = diags([e, -2*e, e], offsets=[-1,0,1], shape=(N,N)).tolil()
L1D[0,-1] = 1
L1D[-1,0] = 1
L1D /= dx**2

# 2D Laplacian operator with periodic BC
L2D = kron(eye(N), L1D) + kron(L1D, eye(N))

# Full Helmholtz operator (laplacian - identity)
A = L2D - eye(N*N)

# Flattened initial phi
phi_vec = phi.ravel()

# Compute initial q = laplacian - phi
q_vec = A @ phi_vec

# Time stepping
t = 0.0
nsteps = int(tmax/dt)
output_times = [0.0, tmax/3, 2*tmax/3, tmax]
phi_snapshots = []
output_idx = 0

while t <= tmax + 1e-8:
    # Reshape phi
    phi_grid = phi_vec.reshape((N,N))
    
    # dphi/dy finite difference with periodic BC
    dphi_dy = np.roll(phi_grid, -1, axis=0) - np.roll(phi_grid, +1, axis=0)
    dphi_dy /= (2*dy)
    
    # RHS of q update
    rhs = -v_star * dphi_dy.ravel()
    
    # Update q
    q_vec = q_vec + dt*rhs
    
    # Solve Helmholtz: A * phi_new = q
    phi_vec, info = linalg.cg(A, q_vec, atol=1e-8, maxiter=500)
    
    if np.isclose(t, output_times[output_idx], atol=dt/2):
        phi_snapshots.append(phi_vec.reshape((N,N)).copy())
        output_idx +=1
        if output_idx >= len(output_times):
            break
    
    t += dt

# Plot snapshots
plt.figure(figsize=(12,4))
for idx, phi_snap in enumerate(phi_snapshots):
    plt.subplot(1,4,idx+1)
    plt.pcolormesh(X,Y,phi_snap,cmap='RdBu', shading='auto')
    plt.title(f't = {output_times[idx]:.2f}')
    
    # Show x and y axes with labels
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Optional: Add a colorbar
    plt.colorbar(label='φ')
    
    # Optional: Set nice tick spacing
    plt.xticks(np.linspace(0, L, 5))
    plt.yticks(np.linspace(0, L, 5))

plt.tight_layout()
plt.show()
