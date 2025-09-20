# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, linalg
import time

# Parameters (updated to match solver_inversion.py)
L = 2*np.pi*10
N = 256
dx = L / N
dy = L / N
dt = 2e1
tmax = 1e4
v_star = 2e-2 # of order e-2

x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)
output_times = [0.0, tmax/3, 2*tmax/3, tmax] 

# Initial condition (updated to match solver_inversion.py)
Dx = 5 # spatial scale of initial condition, should be larger than 1
phi0 = 1e-1 * np.exp(-((X - L/2)**2 + (Y - L/2)**2)/(2*Dx**2)) # monopole magnitude: 1e-1
# phi0 = 1e-1* np.exp(-((X - L/2)**2 + (Y - L/2)**2)/(2*5**2))*((x-L/2)/Dx) # dipole
# phi0 = 1e-1* np.sin(0.2*X) * np.sin(0.3*Y) # sinusoidal
# phi0 = 1e-1* np.sin(0.2*X) * np.exp(-((Y - L/2)**2)/(2*Dx**2)) # sinusoidal in x and gaussian in y
# phi0 = 1e-1* np.exp(-((X - L/2)**2)/(2*Dx**2)) * np.sin(0.2*Y) # gaussian in x and sinusoidal in y

# -------- Method 2: Sparse Matrix Iterative Inversion with RK4 --------
# Run this first to get actual output times
start_time_sparse = time.time()

# Construct Helmholtz operator with perriodic BC
e = np.ones(N)
L1D = diags([e, -2*e, e], offsets=[-1,0,1], shape=(N,N)).tolil()
L1D[0,-1] = 1
L1D[-1,0] = 1
L1D /= dx**2
L2D = kron(eye(N), L1D) + kron(L1D, eye(N))
A_sparse = L2D - eye(N*N)

# Use initial condition directly for t=0 snapshot
phi_vec = phi0.ravel()
q_vec = A_sparse @ phi_vec # q=(\nabla^2-1)\phi
snapshots_sparse = []
snapshots_sparse.append(phi0.copy())  # Store initial condition as first snapshot
output_idx = 1  # Start from second output time

# Define RK4 RHS calculation function
def compute_rhs(q):
    # Solve Helmholtz: A * phi = q
    phi, _ = linalg.cg(A_sparse, q, atol=1e-8, maxiter=500)
    
    # Reshape phi for finite difference
    phi_grid = phi.reshape((N,N))
    
    # dphi/dy finite difference with periodic BC
    dphi_dy = np.roll(phi_grid, -1, axis=0) - np.roll(phi_grid, +1, axis=0)
    dphi_dy /= (2*dy)
    
    # Return RHS
    return v_star * dphi_dy.ravel()

# Time stepping
t = dt  # Start from first time step
actual_snapshot_times = [0.0]  # First snapshot is at t=0

while t <= tmax + 1e-8:
    # RK4 integration step
    r1 = compute_rhs(q_vec)
    r2 = compute_rhs(q_vec + 0.5*dt*r1)
    r3 = compute_rhs(q_vec + 0.5*dt*r2)
    r4 = compute_rhs(q_vec + dt*r3)
    
    # Update q using RK4 formula
    q_vec = q_vec + (dt/6)*(r1 + 2*r2 + 2*r3 + r4)
    
    # Update phi from new q
    phi_vec, info = linalg.cg(A_sparse, q_vec, atol=1e-8, maxiter=500)
    
    if np.isclose(t, output_times[output_idx], atol=dt/2):
        snapshots_sparse.append(phi_vec.reshape((N,N)).copy())
        actual_snapshot_times.append(t)  # Record the actual time
        # print(f"Snapshot taken at t = {t} (target: {output_times[output_idx]})")
        output_idx += 1
        if output_idx >= len(output_times):
            break
    
    t += dt

sparse_time = time.time() - start_time_sparse
print(f"Sparse matrix with RK4 completed in {sparse_time:.2f} seconds")
# %%

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
# print("Evaluating FFT at actual times:", actual_snapshot_times)
for t in actual_snapshot_times:
    phase = np.exp(1j * v_star * KY * t / A_fft)
    phi_hat_t = phi_hat0 * phase
    phi_t = np.real(np.fft.ifft2(phi_hat_t))
    snapshots_fft.append(phi_t)

fft_time = time.time() - start_time_fft
print(f"FFT spectral method completed in {fft_time:.2f} seconds")
print(f"Speed ratio: {sparse_time/fft_time:.1f}x slower than FFT")
# %%

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

    # Second column: Sparse Inversion with RK4
    ax2 = plt.subplot(len(actual_snapshot_times), 3, idx*3+2)
    im2 = plt.pcolormesh(X, Y, phi_sparse, cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
    plt.title(f'Sparse RK4 t={t:.2f}')
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
plt.savefig("spectral_vs_sparse_rk4_actual_times.png", dpi=300)
plt.show()

print("Actual snapshot times:", actual_snapshot_times)
print("L2 norms at each time:", l2_norms)
print("L-infinity norms at each time:", linf_norms)
print(f"Mean L2 norm: {np.mean(l2_norms):.6e}")
print(f"Mean L-infinity norm: {np.mean(linf_norms):.6e}")
# %%
# -------- Power Spectrum Analysis and Visualization --------
def plot_power_spectra(snapshots_fft, snapshots_sparse, actual_snapshot_times):
    """
    Plot 2D power spectrum of phi for both methods at all time points.
    """
    # Declare globals first
    global spectrum_fft, spectrum_sparse, power_fft, power_sparse
    global KX_centered, KY_centered, kx_centered, ky_centered
    global spectrum_fft_all, spectrum_sparse_all, power_fft_all, power_sparse_all, power_diff
    
    # Initialize lists to store spectral data for all times
    spectrum_fft_all = []
    spectrum_sparse_all = []
    power_fft_all = []
    power_sparse_all = []
    plt.figure(figsize=(12, 4*len(actual_snapshot_times)))
    
    for idx, t in enumerate(actual_snapshot_times):
        # Get snapshots at this time
        phi_fft = snapshots_fft[idx]
        phi_sparse = snapshots_sparse[idx]
        
        # Compute 2D FFT and shift zero frequency to center
        spectrum_fft = np.fft.fftshift(np.fft.fft2(phi_fft))
        spectrum_sparse = np.fft.fftshift(np.fft.fft2(phi_sparse))

        # Store spectral data for this time step
        spectrum_fft_all.append(spectrum_fft)
        spectrum_sparse_all.append(spectrum_sparse)
        
        # Compute power spectrum (magnitude squared)
        power_fft = np.abs(spectrum_fft)**2
        power_sparse = np.abs(spectrum_sparse)**2

        # Store power data for this time step
        power_fft_all.append(power_fft)
        power_sparse_all.append(power_sparse)
        
        # Apply log scale for better visualization (add small value to avoid log(0))
        power_fft_log = np.log10(power_fft + 1e-15)
        power_sparse_log = np.log10(power_sparse + 1e-15)
        
        # Calculate difference
        power_diff = power_fft_log - power_sparse_log
        
        # Create centered wavenumber grids for plotting
        kx_centered = np.fft.fftshift(np.fft.fftfreq(N, d=dx)*2*np.pi)
        ky_centered = np.fft.fftshift(np.fft.fftfreq(N, d=dy)*2*np.pi)
        KX_centered, KY_centered = np.meshgrid(kx_centered, ky_centered)
        
        # Determine common color scale for both spectra
        vmin_power = min(np.min(power_fft_log), np.min(power_sparse_log))
        vmax_power = max(np.max(power_fft_log), np.max(power_sparse_log))
        
        # Determine color scale for difference
        abs_max_diff = max(abs(np.min(power_diff)), abs(np.max(power_diff)))
        vmin_diff, vmax_diff = -abs_max_diff, abs_max_diff
        
        # Plot FFT Spectral power
        ax1 = plt.subplot(len(actual_snapshot_times), 3, idx*3+1)
        im1 = plt.pcolormesh(KX_centered, KY_centered, power_fft_log, 
                            cmap='viridis', shading='auto',
                            vmin=vmin_power, vmax=vmax_power)
        plt.title(f'FFT Spectrum t={t:.2f}')
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.colorbar(im1, label='log10(Power)')
        plt.axis('equal')
        
        # Optional: Add circle markers to highlight key wavenumbers
        # circle = plt.Circle((0, 0), 10, fill=False, color='r', linestyle='--')
        # ax1.add_patch(circle)
        
        # Plot Sparse RK4 power
        ax2 = plt.subplot(len(actual_snapshot_times), 3, idx*3+2)
        im2 = plt.pcolormesh(KX_centered, KY_centered, power_sparse_log, 
                            cmap='viridis', shading='auto',
                            vmin=vmin_power, vmax=vmax_power)
        plt.title(f'Sparse RK4 Spectrum t={t:.2f}')
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.colorbar(im2, label='log10(Power)')
        plt.axis('equal')
        
        # Plot difference
        ax3 = plt.subplot(len(actual_snapshot_times), 3, idx*3+3)
        im3 = plt.pcolormesh(KX_centered, KY_centered, power_diff,
                            cmap='RdBu', shading='auto',
                            vmin=vmin_diff, vmax=vmax_diff)
        plt.title(f'Spectrum Difference\nt={t:.2f}')
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.colorbar(im3, label='Δlog10(Power)')
        plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig("power_spectrum_comparison.png", dpi=300)
    plt.show()

# Add after your existing visualization code:
print("\nGenerating power spectrum visualizations...")
plot_power_spectra(snapshots_fft, snapshots_sparse, actual_snapshot_times)


# %%
