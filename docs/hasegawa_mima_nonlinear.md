# Hasegawa-Mima Nonlinear Equation with Pseudo-Spectral Method

## Introduction

This simulation solves the nonlinear Hasegawa-Mima equation for drift wave turbulence in magnetized plasmas, using a pseudo-spectral method with RK4 time integration and 2/3 rule dealiasing:

**Governing equation:**
$$\frac{\partial q}{\partial t} + \left[\{\phi, q\}\right] + v_* \frac{\partial \phi}{\partial y} = 0$$

Where:
$$q = \nabla^2 \phi - \phi$$,
$$\left[\{\phi, q\} \right]= \frac{\partial \phi}{\partial x}\frac{\partial q}{\partial y} - \frac{\partial \phi}{\partial y}\frac{\partial q}{\partial x}$$

**Physical variables:**

- $\phi$ = electrostatic potential
- $q$ = generalized vorticity
- $\{\phi, q\}$ = Poisson bracket (nonlinear advection term)
- $v_*$ = diamagnetic drift velocity (fixed at 0.02)
- Domain is periodic in both x and y directions

### Time Integration

4th-order Runge-Kutta (RK4) method for temporal discretization:

$$q^{n+1} = q^n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

where $k_i$ are the RK4 stage evaluations of the RHS in spectral space.

### Spatial Discretization

Pseudo-spectral method using 2D FFT:

- All spatial derivatives computed via spectral differentiation: $\partial \hat{f}/\partial x= ik_x \hat{f}$
- Nonlinear terms computed in physical space then transformed back
- 2/3 rule dealiasing applied to prevent aliasing errors

### Helmholtz Solver

At each RK4 stage, solve for $\phi$ from $q$ using the Helmholtz relation in spectral space:

$$\hat{\phi} = \frac{\hat{q}}{-(k_x^2 + k_y^2 + 1)}$$

This inversion is exact and computationally cheap in Fourier space.

### Dealiasing

The 2/3 rule dealiasing mask is applied to nonlinear Poisson bracket terms:

- Keep modes with $|k_x| \leq N/3$ and $|k_y| \leq N/3$
- Zero out higher wavenumbers before transforming back
- Prevents aliasing errors from quadratic nonlinearity

## Test Cases

The case key in the config file sets different initial conditions:

1. **monopole** - Gaussian monopole centered in domain:
   - $\phi_0 = 0.1 \exp\left(-\frac{(x-L/2)^2 + (y-L/2)^2}{2D_x^2}\right)$

2. **dipole** - Gaussian dipole (odd in x):
   - $\phi_0 = 0.1 \exp\left(-\frac{(x-L/2)^2 + (y-L/2)^2}{2D_x^2}\right) \cdot \frac{x-L/2}{D_x}$

3. **sinusoidal** - Pure sinusoidal in both directions:
   - $\phi_0 = 0.1 \sin(0.2x) \sin(0.3y)$

4. **sin_x_gauss_y** - Sinusoidal in x, Gaussian in y:
   - $\phi_0 = 0.1 \sin(0.2x) \exp\left(-\frac{(y-L/2)^2}{2D_x^2}\right)$

5. **gauss_x_sin_y** - Gaussian in x, sinusoidal in y:
   - $\phi_0 = 0.1 \exp\left(-\frac{(x-L/2)^2}{2D_x^2}\right) \sin(0.2y)$

The simulated results are considered correct if the L2 RMSE (comparing with higher resolution) meets the precision-dependent tolerance (low: 0.0005, medium: 0.0001, high: 0.00001).

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **N Spatial Grid Number (iterative)**
   - N is the grid resolution: $\Delta x = \Delta y = L / N$, where $L$ is domain size

2. **dt Time Step Size (iterative)**
   - dt is the time step for RK4 integration

### Dummy Strategy

1. **N Convergence Search (iterative)**
   - For dummy solution, this means doubling N each iteration (multiplication factor: 2) starting from 32 until convergence
   - **Non-target parameters**: dt∈{5.0, 10.0, 20.0, 40.0}

2. **dt Convergence Search (iterative)**
   - For dummy solution, this means halving dt each iteration (multiplication factor: 0.5) starting from 40.0 until convergence
   - **Non-target parameters**: N∈{32, 64, 128, 256}

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| N | Grid resolution (number of grid points in each direction) | 32 ≤ N ≤ 256 |
| dt | Time step for RK4 integration | 5.0 ≤ dt ≤ 40.0 |

More Notes:

- Smaller N → coarser grid → lower accuracy but lower cost
- Larger dt → fewer time steps → risk of instability but lower cost
- Nonlinear solver uses resolution convergence checking (no analytical solution available)
- N determines spatial resolution: $\Delta x = \Delta y = L / N$ (larger N = finer grid = higher accuracy but higher cost)
- dealias_ratio fixed at 2/3 for stability (dealiases nonlinear Poisson bracket terms)

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| L | Domain size | 62.83185307179586 (2π × 10) |
| v_star | Diamagnetic drift velocity | 0.02 |
| Dx | Initial condition spatial scale | 5.0 |
| dealias_ratio | Dealiasing ratio for 2/3 rule | 0.6667 (2/3) |
| case | Initial condition type | "monopole" |
| record_dt | Time interval between recordings | 100.0 |
| end_frame | Simulation end after certain number of frames | 10 |
| dump_dir | Directory for output files | "sim_res/hasegawa_mima_nonlinear/p1" |
| verbose | Enable verbose output | false |

## Checkout

### Summary

- **Benchmarks**:
  - **p1**: Monopole (Gaussian blob centered in domain)
  - **p2**: Dipole (Gaussian dipole initial condition)
  - **p3**: Sinusoidal (pure sinusoidal in both directions)
  - **p4**: sin_x_gauss_y (sinusoidal in x, Gaussian in y)
  - **p5**: gauss_x_sin_y (Gaussian in x, sinusoidal in y)
- **Target Parameters**: 2 (N, dt)
- **Precision Levels**: 3 (low: 0.0005, medium: 0.0001, high: 0.00001)

### Task Distribution

Current configuration generates:

- **N** (iterative): 5 profiles × 4 non-target combos (4 dt values) = 20 tasks
- **dt** (iterative): 5 profiles × 4 non-target combos (4 N values) = 20 tasks
- **Total per precision**: 40 tasks
- **Total tasks**: 120 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/hasegawa_mima_nonlinear.yaml`
Cache script: `checkouts/hasegawa_mima_nonlinear.py`
