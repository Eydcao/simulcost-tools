import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 2 * np.pi
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
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
k2 = KX ** 2 + KY ** 2
A = -(1 + k2)

# Initial condition
phi0 = np.exp(-((X - L/2)**2 + (Y - L/2)**2) / (0.2)**2)
# phi0 = np.sin(2*X) * np.sin(3*Y)

# --- SPECTRAL TIME INTEGRATION ---
phi_hat0 = np.fft.fft2(phi0)

# --- FINITE DIFFERENCE TIME STEPPING ---
phi_fd = phi0.copy()
t = 0.0
nsteps = int(tmax / dt)

# Prepare arrays to store all frames for animation
frames_spectral = []
frames_fd = []

for step in range(nsteps + 1):
    # Spectral solution at this time
    phase = np.exp(-1j * v_star * KY * t / A)
    phi_hat_t = phi_hat0 * phase
    phi_t = np.real(np.fft.ifft2(phi_hat_t))
    frames_spectral.append(phi_t)

    # Finite difference step
    phi_hat = np.fft.fft2(phi_fd)
    lap_phi_hat = -k2 * phi_hat
    lap_phi = np.real(np.fft.ifft2(lap_phi_hat))
    q = lap_phi - phi_fd

    dphi_dy = np.roll(phi_fd, -1, axis=0) - np.roll(phi_fd, +1, axis=0)
    dphi_dy /= (2 * dy)

    q_new = q - dt * v_star * dphi_dy
    q_new_hat = np.fft.fft2(q_new)
    phi_hat_new = q_new_hat / A
    phi_fd = np.real(np.fft.ifft2(phi_hat_new))
    frames_fd.append(phi_fd)

    t += dt

# --- ANIMATION ---

vmin = min(np.min(f) for f in frames_spectral + frames_fd)
vmax = max(np.max(f) for f in frames_spectral + frames_fd)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
ax1, ax2 = axes

pcm1 = ax1.pcolormesh(X, Y, frames_spectral[0], cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
ax1.set_title("Spectral time integration")
# ax1.axis('off')  # Remove this line to show axes
pcm2 = ax2.pcolormesh(X, Y, frames_fd[0], cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
ax2.set_title("Finite difference time stepping")
# ax2.axis('off')  # Remove this line to show axes

ax1.set_xlabel('x')
ax1.set_ylabel('y')

cbar = fig.colorbar(pcm2, ax=axes, location='right', shrink=0.8, label='Field value', pad=0.02)

def animate(i):
    pcm1.set_array(frames_spectral[i].ravel())
    pcm2.set_array(frames_fd[i].ravel())
    ax1.set_title(f"Spectral time integration\nt={i*dt:.2f}")
    ax2.set_title(f"Finite difference time stepping\nt={i*dt:.2f}")
    return pcm1, pcm2

ani = animation.FuncAnimation(
    fig, animate, frames=len(frames_spectral), interval=10, blit=False  # interval reduced from 30 to 10 ms
)

plt.show()