# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq, fftshift
import time

# ---------------- Parameters ----------------
N = 128                # grid points
L = 2 * np.pi * 10      # domain size
dt = 1e0               # time step
tmax = 1e4             # max time
v_star = 2e-2            # diamagnetic drift velocity, of order e-2

dealias_ratio = 2/3 # stronger dealiasing with smaller dealias_ratio

# ---------------- Grid ----------------
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

dx = dy = L / N
kx = fftfreq(N, d=dx) * 2 * np.pi
ky = fftfreq(N, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
k2 = KX**2 + KY**2
A_fft = -(1 + k2)

# Define output times to match solver_comparison
output_times = [0.0, tmax/3, 2*tmax/3, tmax]

# ---------------- Dealiasing Mask ----------------
def dealias_mask(N):
    cutoff = int(N * dealias_ratio // 2)
    mask = np.zeros(N)
    mask[:cutoff+1] = 1
    mask[-cutoff:] = 1
    return mask

mask_x = dealias_mask(N)
mask_y = dealias_mask(N)
dealias = np.outer(mask_y, mask_x)

# ---------------- Initial Condition ----------------
Dx = 5 # spatial scale of initial condition # should be larger than 1
# phi0 = 1e0 * np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (2*Dx**2)) #monopole, magnitude: 1e-1
# phi0 = 1e0* np.exp(-((X - L/2)**2 + (Y - L/2)**2)/(2*5**2))*((x-L/2)/Dx) # dipole
# phi0 = 1e0* np.sin(0.2*X) * np.sin(0.3*Y) # sinusoidal
# phi0 = 1e0* np.sin(0.2*X) * np.exp(-((Y - L/2)**2)/(2*Dx**2)) # sinusoidal in x and gaussian in y
phi0 = 1e0* np.exp(-((X - L/2)**2)/(2*Dx**2)) * np.sin(0.2*Y) # gaussian in x and sinusoidal in y

phi0_hat = fft2(phi0) 
q_hat = A_fft * phi0_hat  # Poisson equation in spectral space # q=(\nabla^2-1)\phi

def poisson_bracket_dealiased(phi_hat, q_hat):
    """
    Compute {phi, q} using spectral method with proper dealiasing
    Applies the 2/3 rule dealiasing only to the nonlinear Poisson bracket
    """
    
    # Compute derivatives in spectral space
    dphidx_hat = 1j * KX * phi_hat
    dphidy_hat = 1j * KY * phi_hat
    dqdx_hat = 1j * KX * q_hat
    dqdy_hat = 1j * KY * q_hat

    # Transform to physical space
    dphidx = ifft2(dphidx_hat).real
    dphidy = ifft2(dphidy_hat).real
    dqdx = ifft2(dqdx_hat).real
    dqdy = ifft2(dqdy_hat).real
    
    # Compute Jacobian in physical space
    jacobian = dphidx * dqdy - dphidy * dqdx
    
    # Transform back to spectral space with dealiasing
    jacobian_hat = fft2(jacobian) * dealias
    
    return jacobian_hat

def rhs(q_hat):
    # Compute phi from q using Poisson equation in spectral space
    phi_hat = q_hat / A_fft  # No division by zero issue at k=0
    
    # Compute dealiased Jacobian in spectral space
    jacobian_hat = poisson_bracket_dealiased(phi_hat, q_hat)
    
    # Compute y derivative (linear term, no dealiasing needed)
    dphidy_hat = 1j * KY * phi_hat
    
    # Compute RHS in spectral space
    rhs_hat = -jacobian_hat + v_star * dphidy_hat
    
    return rhs_hat

# ---------------- Nonlinear HM Solver (RK4) ----------------
start_time_nonlinear = time.time()

# Store snapshots at specific output times
snapshots_nonlinear = []
actual_snapshot_times = []

# Store initial condition directly
snapshots_nonlinear.append(phi0.copy())
actual_snapshot_times.append(0.0)

t = 0.0
output_idx = 1  # Start from second output time

while t <= tmax + 1e-8 and output_idx < len(output_times):
    # RK4 integration step
    r1 = rhs(q_hat)
    r2 = rhs(q_hat + 0.5 * dt * r1)
    r3 = rhs(q_hat + 0.5 * dt * r2)
    r4 = rhs(q_hat + dt * r3)
    q_hat += dt / 6 * (r1 + 2 * r2 + 2 * r3 + r4)
    t += dt
    print(t, end='\r')  # Print current time for progress tracking
    
    # Check if we're close to an output time
    if np.isclose(t, output_times[output_idx], atol=dt/2):
        phi_hat = q_hat / A_fft
        phi = ifft2(phi_hat).real
        snapshots_nonlinear.append(phi.copy())
        actual_snapshot_times.append(t)
        # print(f"Nonlinear snapshot taken at t = {t:.2f} (target: {output_times[output_idx]:.2f})")
        output_idx += 1

nonlinear_time = time.time() - start_time_nonlinear
print(f"Nonlinear HM solver completed in {nonlinear_time:.2f} seconds")
# %%

# ---------------- Linear FFT Spectral Method ----------------
# Now run linear spectral method using the actual times from nonlinear method
start_time_fft = time.time()

snapshots_fft = []
for t in actual_snapshot_times:
    phase = np.exp(1j * v_star * KY * t / A_fft)
    phi_hat_t = phi0_hat * phase  # Using phi0_hat directly (same as phi_hat0)
    phi_t = np.real(np.fft.ifft2(phi_hat_t))
    snapshots_fft.append(phi_t)

fft_time = time.time() - start_time_fft
print(f"Linear FFT spectral method completed in {fft_time:.2f} seconds")
print(f"Speed ratio: {nonlinear_time/fft_time:.1f}x slower than linear FFT")
# %%

# ---------------- Visualization ----------------
# Calculate min and max for consistent colormap
vmin = min(np.min(s) for s in snapshots_fft + snapshots_nonlinear)
vmax = max(np.max(s) for s in snapshots_fft + snapshots_nonlinear)
# Make the colormap symmetric around zero
abs_max = max(abs(vmin), abs(vmax))
vmin, vmax = -abs_max, abs_max

plt.figure(figsize=(12, 4*len(actual_snapshot_times)))
for idx, t in enumerate(actual_snapshot_times):
    phi_fft = snapshots_fft[idx]
    phi_nonlinear = snapshots_nonlinear[idx]

    # First column: Linear FFT Spectral
    ax1 = plt.subplot(len(actual_snapshot_times), 2, idx*2+1)
    im1 = plt.pcolormesh(X, Y, phi_fft, cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
    plt.title(f'Linear FFT t={t:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(im1, label='φ')

    # Second column: Nonlinear HM
    ax2 = plt.subplot(len(actual_snapshot_times), 2, idx*2+2)
    im2 = plt.pcolormesh(X, Y, phi_nonlinear, cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
    plt.title(f'Nonlinear HM t={t:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(im2, label='φ')

plt.tight_layout()
plt.savefig("linear_vs_nonlinear_hm_comparison.png", dpi=300)
plt.show()

print("Actual snapshot times:", actual_snapshot_times)
# %%
