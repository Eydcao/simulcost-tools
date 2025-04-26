import os
import numpy as np
from diff_react import DiffReac
import h5py

# Grid parameters
L = 512.0  # domain length
nx = int(L * 4)
dx = L / nx  # spatial step
x = np.linspace(0, L, nx)  # grid points (left of each cell)

# Time parameters
T = 256  # end time
dt = dx / 2
nt = int(T / dt)  # number of time steps
record_dt = 2
record_steps = int(record_dt / dt)

# Initial condition
# (0~2) is 1 (2~) is 0
u_0 = np.where((x >= 0) & (x <= 2), 1.0, 0.0)

# Create Optimizer
solver = DiffReac(tol=1e-9, max_iter=2000, min_step=1e-3, verbose=False)

# Time stepping
u = u_0.copy()  # current solution
solutions = [u_0.copy()]  # store solutions for plotting

for n in range(nt):
    # Use current solution as initial guess for next time step
    u_guess = u.copy()

    # Solve for next time step
    u_next, success, iters, residual_norm = solver.optimize(u_guess, dt=dt, dx=dx, u_0=u)

    if not success:
        print(f"Failed to converge at time step {n+1}")
    #     break # TODO uncomment to stop at first failure

    print(f"Time step {n+1}/{nt} completed in {iters} iterations, residual norm: {residual_norm:.2e}")

    # Update solution
    u = u_next

    # Append to recordings
    if (n + 1) % record_steps == 0:
        solutions.append(u.copy())

# Make sols a numpy array
solutions = np.array(solutions)

# Create directory if it does not exist
output_dir = f"res_dx_{dx}_dt_{dt}_record_dt_{record_dt}_endT_{T}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the results as an HDF5 file
with h5py.File(os.path.join(output_dir, "solutions.h5"), "w") as f:
    f.create_dataset("solutions", data=solutions)
    f.create_dataset("x", data=x)
    f.create_dataset("u_0", data=u_0)
    f.create_dataset("dx", data=dx)
    f.create_dataset("dt", data=dt)
    f.create_dataset("record_dt", data=record_dt)
