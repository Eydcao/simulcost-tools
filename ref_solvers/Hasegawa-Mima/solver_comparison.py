import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2*np.pi
N = 256
dx = L / N
dy = L / N
dt = 0.01
tmax = 5.0
v_star = 1.0

# Grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Wavenumbers for FFT
kx = np.fft.fftfreq(N, d=dx)*2*np.pi
ky = np.fft.fftfreq(N, d=dy)*2*np.pi
KX, KY = np.meshgrid(kx, ky)
k2 = KX**2 + KY**2
A = -(1 + k2)

# Initial condition
phi0 = np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (1)**2)
# phi0 = np.sin(2*X) * np.sin(3*Y)
# phi0 = np.sin(1.3*X) * np.sin(1.7*Y)

# Time array
times = [0.0, tmax/3, 2*tmax/3, tmax]

# Containers for snapshots
snapshots_spectral = []
snapshots_fd = []

# --- SPECTRAL TIME INTEGRATION ---
phi_hat0 = np.fft.fft2(phi0)

for t in times:
    phase = np.exp(-1j * v_star * KY * t / A)
    phi_hat_t = phi_hat0 * phase
    phi_t = np.real(np.fft.ifft2(phi_hat_t))
    snapshots_spectral.append(phi_t)

# --- FINITE DIFFERENCE TIME STEPPING ---
phi_fd = phi0.copy()
t = 0.0
nsteps = int(tmax / dt)
output_idx = 0

while t <= tmax + 1e-8:
    # Compute q = laplacian - phi
    phi_hat = np.fft.fft2(phi_fd)
    lap_phi_hat = -k2 * phi_hat
    lap_phi = np.real(np.fft.ifft2(lap_phi_hat))
    q = lap_phi - phi_fd

    # dphi/dy by finite difference
    dphi_dy = np.roll(phi_fd, -1, axis=0) - np.roll(phi_fd, +1, axis=0)
    dphi_dy /= (2*dy)

    # Forward Euler update
    q_new = q - dt * v_star * dphi_dy

    # Solve Helmholtz equation in Fourier space
    q_new_hat = np.fft.fft2(q_new)
    phi_hat_new = q_new_hat / A
    phi_fd = np.real(np.fft.ifft2(phi_hat_new))

    # Save snapshots at the same times
    if np.isclose(t, times[output_idx], atol=dt/2):
        snapshots_fd.append(phi_fd.copy())
        output_idx += 1
        if output_idx >= len(times):
            break

    t += dt

# --- PLOT RESULTS ---




# Use subplots for better layout and automatic colorbar placement
# Set a smaller and wider figure size for better fit on screen
fig, axes = plt.subplots(len(times), 2, figsize=(7, 4.5), constrained_layout=True)

# Try to auto-center and resize the window if possible (works with some backends)
try:
    mng = plt.get_current_fig_manager()
    mng.resize(400, 600)
    mng.window.wm_geometry("+100+100")
except Exception:
    pass



# Set common color limits for all plots
vmin = min(np.min(snap) for snap in snapshots_spectral + snapshots_fd)
vmax = max(np.max(snap) for snap in snapshots_spectral + snapshots_fd)


for idx, t in enumerate(times):
    ax1 = axes[idx, 0]
    ax2 = axes[idx, 1]
    pcm1 = ax1.pcolormesh(X, Y, snapshots_spectral[idx], cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
    ax1.set_title(f"Spectral time integration\nt={t:.2f}")
    ax1.axis('off')
    pcm2 = ax2.pcolormesh(X, Y, snapshots_fd[idx], cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
    ax2.set_title(f"Finite difference time stepping\nt={t:.2f}")
    ax2.axis('off')

# Add a single colorbar for all plots, placed to the right
cbar = fig.colorbar(pcm2, ax=axes, location='right', shrink=0.8, label='Field value', pad=0.02)
plt.show()

# --- Compute and print error norms between methods ---

l2_norms = []
linf_norms = []

for idx in range(len(times)):
    spectral = snapshots_spectral[idx]
    fd = snapshots_fd[idx]
    diff = spectral - fd
    l2 = np.sqrt(np.mean(diff**2))
    linf = np.max(np.abs(diff))
    l2_norms.append(l2)
    linf_norms.append(linf)

mean_l2 = np.mean(l2_norms)
mean_linf = np.mean(linf_norms)

print(f"Mean L2 norm between methods: {mean_l2:.6e}")
print(f"Mean L-infinity norm between methods: {mean_linf:.6e}")
