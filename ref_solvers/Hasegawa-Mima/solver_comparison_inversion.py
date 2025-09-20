import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, linalg
import time

# Parameters (updated to match solver_inversion.py)
L = 2*np.pi*10
N = 256
dx = L / N
dy = L / N
dt = 5
tmax = 1e4
v_star = 2e-2

x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)
output_times = [0.0, tmax/3, 2*tmax/3, tmax]

# Initial condition (updated to match solver_inversion.py)
phi0 = 2* np.exp(-((X - L/2)**2 + (Y - L/2)**2)/(2*5**2))

# -------- Method 2: Sparse Matrix Iterative Inversion --------
# Run this first to get actual output times
start_time_sparse = time.time()

# Construct Helmholtz operator
e = np.ones(N)
L1D = diags([e, -2*e, e], offsets=[-1,0,1], shape=(N,N)).tolil()
L1D[0,-1] = 1
L1D[-1,0] = 1
L1D /= dx**2
L2D = kron(eye(N), L1D) + kron(L1D, eye(N))
A_sparse = L2D - eye(N*N)

# Use initial condition directly for t=0 snapshot
phi_vec = phi0.ravel()
q_vec = A_sparse @ phi_vec
snapshots_sparse = []
snapshots_sparse.append(phi0.copy())  # Store initial condition as first snapshot
output_idx = 1  # Start from second output time
actual_snapshot_times = [0.0]  # Record the actual snapshot times

# Time stepping
t = dt  # Start from first time step (skip t=0)
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
    phi_vec, info = linalg.cg(A_sparse, q_vec, atol=1e-8, maxiter=500)
    
    if np.isclose(t, output_times[output_idx], atol=dt/2):
        snapshots_sparse.append(phi_vec.reshape((N,N)).copy())
        actual_snapshot_times.append(t)  # Record the actual time
        output_idx += 1
        if output_idx >= len(output_times):
            break
    
    t += dt

sparse_time = time.time() - start_time_sparse
print(f"Sparse matrix method completed in {sparse_time:.2f} seconds")

# -------- Method 1: FFT Spectral Method --------
# Now run spectral method using the actual times from sparse method
start_time_fft = time.time()

phi_hat0 = np.fft.fft2(phi0)
kx = np.fft.fftfreq(N, d=dx)*2*np.pi
ky = np.fft.fftfreq(N, d=dy)*2*np.pi
KX, KY = np.meshgrid(kx, ky)
k2 = KX**2 + KY**2
A_fft = -(1 + k2)

snapshots_fft = []
print("Evaluating FFT at actual times:", actual_snapshot_times)
for t in actual_snapshot_times:
    phase = np.exp(-1j * v_star * KY * t / A_fft)
    phi_hat_t = phi_hat0 * phase
    phi_t = np.real(np.fft.ifft2(phi_hat_t))
    snapshots_fft.append(phi_t)

fft_time = time.time() - start_time_fft
print(f"FFT spectral method completed in {fft_time:.2f} seconds")
print(f"Speed ratio: {sparse_time/fft_time:.1f}x slower than FFT")

# -------- Error Calculation and Visualization --------
vmin = min(np.min(s) for s in snapshots_fft + snapshots_sparse)
vmax = max(np.max(s) for s in snapshots_fft + snapshots_sparse)
# Make the colormap symmetric around zero
abs_max = max(abs(vmin), abs(vmax))
vmin, vmax = -abs_max, abs_max

# Calculate min and max for difference plots to use consistent colormap
diff_snapshots = [snapshots_fft[i] - snapshots_sparse[i] for i in range(len(actual_snapshot_times))]
diff_min = min(np.min(d) for d in diff_snapshots)
diff_max = max(np.max(d) for d in diff_snapshots)
# Make the colormap symmetric around zero
diff_abs_max = max(abs(diff_min), abs(diff_max))
diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max

l2_norms = []
linf_norms = []

plt.figure(figsize=(12, 4*len(actual_snapshot_times)))
for idx, t in enumerate(actual_snapshot_times):
    phi_fft = snapshots_fft[idx]
    phi_sparse = snapshots_sparse[idx]
    # Use the pre-computed difference instead of recalculating
    diff = diff_snapshots[idx]
    l2 = np.sqrt(np.mean(diff**2))
    linf = np.max(np.abs(diff))
    l2_norms.append(l2)
    linf_norms.append(linf)

    # First column: FFT Spectral
    ax1 = plt.subplot(len(actual_snapshot_times), 3, idx*3+1)
    im1 = plt.pcolormesh(X, Y, phi_fft, cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
    plt.title(f'FFT Spectral t={t:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Add colorbar for FFT plot in each row
    plt.colorbar(im1, label='φ')

    # Second column: Sparse Inversion
    ax2 = plt.subplot(len(actual_snapshot_times), 3, idx*3+2)
    im2 = plt.pcolormesh(X, Y, phi_sparse, cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
    plt.title(f'Sparse Inversion t={t:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')

    # Add colorbar for Sparse Matrix plot in each row
    plt.colorbar(im2, label='φ')

    # Third column: Difference
    ax3 = plt.subplot(len(actual_snapshot_times), 3, idx*3+3)
    im3 = plt.pcolormesh(X, Y, diff, cmap='bwr', shading='auto', 
                         vmin=diff_vmin, vmax=diff_vmax)
    plt.title(f'Difference\nL2={l2:.2e}, L∞={linf:.2e}')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Add colorbar for difference plot in each row
    plt.colorbar(im3, label='Δφ')

plt.tight_layout()
plt.savefig("spectral_vs_sparse_aligned_times.png", dpi=300)
plt.show()

print("Actual snapshot times:", actual_snapshot_times)
print("L2 norms at each time:", l2_norms)
print("L-infinity norms at each time:", linf_norms)
print(f"Mean L2 norm: {np.mean(l2_norms):.6e}")
print(f"Mean L-infinity norm: {np.mean(linf_norms):.6e}")