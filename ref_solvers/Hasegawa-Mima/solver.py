import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
L = 2*np.pi         # domain size
Nx = 256           # grid points in x
Ny = 256           # grid points in y
dx = L / Nx
dy = L / Ny
dt = 0.01           # time step
tmax = 10.0          # final time
v_star = 2.0        # diamagnetic drift velocity

# Grid
x = np.linspace(0, L, Nx, endpoint=False)
y = np.linspace(0, L, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

# Initial condition: Gaussian blob
# phi = np.exp(-((X - L/2)**2 + (Y - L/2)**2) / 0.2**2)
# phi = np.sin(1*X) * ((np.abs(Y - L/2) < 0.1).astype(float))
phi = np.sin(3.2*X) * np.sin(2.7*Y)

# Precompute wavenumbers for FFT inversion
kx = np.fft.fftfreq(Nx, d=dx)*2*np.pi
ky = np.fft.fftfreq(Ny, d=dy)*2*np.pi
KX, KY = np.meshgrid(kx, ky)
k2 = KX**2 + KY**2

# Prepare figure
fig, ax = plt.subplots()
fig.suptitle("Time Evolution of the Field")  # <-- static title at the top
cax = ax.pcolormesh(X, Y, phi, cmap='RdBu', shading='auto')
fig.colorbar(cax)
title = ax.set_title("")

# Simulation: store all phi snapshots
nframes = int(tmax / dt) + 1
phi_snapshots = []
# phi = np.exp(-((X - L/2)**2 + (Y - L/2)**2) / 0.2**2)  # reset initial condition

for frame in range(nframes):
    t = frame * dt

    # Compute q = laplacian(phi) - phi
    phi_hat = np.fft.fft2(phi)
    lap_phi_hat = -k2 * phi_hat
    lap_phi = np.real(np.fft.ifft2(lap_phi_hat))
    q = lap_phi - phi

    # Compute dphi/dy by finite difference (periodic)
    dphi_dy = np.roll(phi, -1, axis=0) - np.roll(phi, +1, axis=0)
    dphi_dy /= (2*dy)

    # Update q in time
    q_new = q - dt * v_star * dphi_dy

    # Solve Helmholtz equation: (laplacian - I) phi_new = q_new
    q_new_hat = np.fft.fft2(q_new)
    denom = -(k2 + 1)
    phi_hat_new = q_new_hat / denom
    phi = np.real(np.fft.ifft2(phi_hat_new))

    # Save snapshot
    phi_snapshots.append(phi.copy())

# Only display every 100th frame (plus last frame if not included)
display_frames = list(range(0, nframes, 10))
if display_frames[-1] != nframes - 1:
    display_frames.append(nframes - 1)

def update_anim(i):
    phi = phi_snapshots[display_frames[i]]
    cax.set_array(phi.ravel())
    t = display_frames[i] * dt
    title.set_text(f"t = {t:.2f}")
    return cax, title

anim = FuncAnimation(
    fig, update_anim, frames=len(display_frames), interval=10, blit=False, repeat=False
)

plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
