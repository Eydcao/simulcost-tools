# Hasegawa-Mima Linear Equation with RK4 and CG Solver

## Introduction

This simulation solves the linearized Hasegawa-Mima equation for drift wave dynamics in magnetized plasmas, using 4th-order Runge-Kutta time integration with a Conjugate Gradient solver for the Helmholtz equation:

**Governing equation:**
$$\frac{\partial q}{\partial t} + v_* \frac{\partial \phi}{\partial y} = 0$$

Where:
$$q = \nabla^2 \phi - \phi$$

**Physical variables:**

- $\phi$ = electrostatic potential
- $q$ = generalized vorticity
- $v_*$ = diamagnetic drift velocity
- Domain is periodic in both x and y directions

### Time Integration

4th-order Runge-Kutta (RK4) method for temporal discretization:

$$q^{n+1} = q^n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

where $k_i$ are the RK4 stage evaluations of the RHS.

### Spatial Discretization

Finite differences with periodic boundary conditions on a uniform 2D grid:

- Grid spacing: $\Delta x = \Delta y = L / N$
- 2D Laplacian operator: $\nabla^2 \phi$ discretized using 5-point stencil
- Derivative operator: $\partial \phi / \partial y$ using central differences

### Helmholtz Solver

At each RK4 stage, solve the Helmholtz equation for $\phi$ given $q$:

$$(\nabla^2 - I)\phi = q$$

Solved using sparse Conjugate Gradient (CG) method with:

- **cg_atol**: Absolute tolerance for convergence, which is a tunnable parameter
- **cg_maxiter**: Maximum CG iterations (fixed to be: 1000)

### Analytical Solution

For validation, an exact spectral solution via 2D FFT is available:

$$\phi(t) = \mathcal{F}^{-1}\left[\hat{\phi}_0 \exp\left(i \frac{v_* k_y t}{1 + k^2}\right)\right]$$

where $\mathcal{F}$ denotes Fourier transform and $k^2 = k_x^2 + k_y^2$.

## Test Cases

The case key in the config file sets different initial conditions:

1. **monopole** - Gaussian monopole centered in domain:
   - $\phi_0 = 0.1 \exp\left(-\frac{(x-L/2)^2 + (y-L/2)^2}{2D_x^2}\right)$

2. **dipole** - Gaussian dipole (odd in x):
   - $\phi_0 = 0.1 \exp\left(-\frac{(x-L/2)^2 + (y-L/2)^2}{2D_x^2}\right) \cdot \frac{x-L/2}{D_x}$

3. **sin_x_gauss_y** - Sinusoidal in x, Gaussian in y:
   - $\phi_0 = 0.1 \sin(0.2x) \exp\left(-\frac{(y-L/2)^2}{2D_x^2}\right)$

4. **gauss_x_sin_y** - Gaussian in x, sinusoidal in y:
   - $\phi_0 = 0.1 \exp\left(-\frac{(x-L/2)^2}{2D_x^2}\right) \sin(0.2y)$

The simulated results are considered correct if the L2 RMSE meets the precision-dependent tolerance (low: 0.01, medium: 0.001, high: 0.0005) compared to the analytical solution.

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **N Spatial Grid Number (iterative+0-shot)**
   - N is the grid resolution: $\Delta x = \Delta y = L / N$, where $L$ is domain size

2. **dt Time Step Size (iterative+0-shot)**
   - dt is the time step for RK4 integration

3. **cg_atol (iterative+0-shot)**
   - The absolute residual threshold for the CG solver

### Dummy Strategy

1. **N Convergence Search (iterative+0-shot)**
   - For dummy solution, this means doubling N each iteration (multiplication factor: 2) starting from 32 until convergence
   - **Non-target parameters**: dt∈{5.0, 10.0, 20.0, 40.0}, cg_atol∈{1e0, 1e-1, 1e-2, 1e-3, 1e-4}

2. **dt Convergence Search (iterative+0-shot)**
   - For dummy solution, this means halving dt each iteration (multiplication factor: 0.5) starting from 40.0 until convergence
   - **Non-target parameters**: N∈{32, 64, 128, 256}, cg_atol∈{1e0, 1e-1, 1e-2, 1e-3, 1e-4}

3. **cg_atol Optimization (iterative+0-shot)**
   - For dummy solution, iteratively search from coarse (relaxed) to fine (strict) tolerance, stopping at the first convergent solution
   - Search follows a logarithmic progression through predefined cg_atol values until convergence is achieved
   - **Non-target parameters**: N∈{32, 64, 128, 256}, dt∈{5.0, 10.0, 20.0, 40.0}

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| N | Grid resolution (number of grid points in each direction) | 32 ≤ N ≤ 256 |
| dt | Time step for RK4 integration | 5.0 ≤ dt ≤ 40.0 |
| cg_atol | Conjugate Gradient solver absolute tolerance | 1e-4 ≤ cg_atol ≤ 1e0 |

More Notes:

- Smaller N → coarser grid → lower accuracy but lower cost
- Larger dt → fewer time steps → lower accuracy but lower cost
- Larger cg_atol → fewer CG iterations → lower accuracy but lower cost
- cg_atol = 1e0 typically gives very inaccurate results due to loose CG convergence
- N determines spatial resolution: $\Delta x = L / N$ (larger N = finer grid = higher accuracy but higher cost)

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| L | Domain size | 62.83185307179586 (2π × 10) |
| v_star | Diamagnetic drift velocity | 0.02 |
| Dx | Initial condition spatial scale | 5.0 |
| cg_maxiter | CG solver maximum iterations | 1000 |
| case | Initial condition type | "monopole" |
| analytical | Use analytical solution (true) or numerical (false) | false |
| record_dt | Time interval between recordings | 1000.0 |
| end_frame | Simulation end after certain number of frames | 10 |
| dump_dir | Directory for output files | "sim_res/hasegawa_mima_linear/p1" |
| verbose | Enable verbose output | false |

## Checkout

### Summary

- **Benchmarks**:
  - **p1**: Monopole (Gaussian blob centered in domain)
  - **p2**: Dipole (Gaussian dipole initial condition)
  - **p3**: sin_x_gauss_y (sinusoidal in x, Gaussian in y)
  - **p4**: gauss_x_sin_y (Gaussian in x, sinusoidal in y)
- **Target Parameters**: 3 (N, dt, cg_atol)
- **Precision Levels**: 3 (low: 0.01, medium: 0.001, high: 0.0005)

### Task Distribution

Current configuration generates:

- **N** (iterative+0-shot): 4 profiles × 20 non-target combos = 80 tasks
- **dt** (iterative+0-shot): 4 profiles × 20 non-target combos = 80 tasks
- **cg_atol** (iterative+0-shot): 4 profiles × 16 non-target combos = 64 tasks
- **Total per precision**: 224 tasks
- **Total tasks**: 672 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/hasegawa_mima_linear.yaml`
Cache script: `checkouts/hasegawa_mima_linear.py`
