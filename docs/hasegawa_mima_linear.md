# Hasegawa-Mima Linear Equation with RK4 and CG Solver

## Introduction

This simulation solves the linearized Hasegawa-Mima equation for drift wave dynamics in magnetized plasmas, using 4th-order Runge-Kutta time integration with a Conjugate Gradient solver for the Helmholtz equation.

**Wall Time Constraint**: To prevent runaway simulations, a configurable wall time limit (default: 120 seconds) is enforced. Simulations that exceed this limit are terminated early and flagged as incomplete via the function call.

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

**Simulation Duration:**

- Recording interval: `record_dt = 1000.0` time units
- Number of frames: `end_frame = 10`
- Total simulation time: $T = \text{record\_dt} \times \text{end\_frame} = 10{,}000$ time units
- Number of time steps: $N_{\text{steps}} = T / \Delta t$

### Spatial Discretization

Finite differences with periodic boundary conditions on a uniform 2D grid:

- Domain size: $L = 2\pi \times 10 \approx 62.83$ (periodic in both x and y)
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

### Convergence Method

Convergence is verified by comparing numerical solution against the analytical (spectral) solution:

- Run numerical simulation with given parameters (N, dt, cg_atol)
- Run analytical solution using 2D FFT (exact, serves as reference)
- Calculate L2 RMSE between numerical and analytical solutions across all frames
- Converged if RMSE < tolerance threshold

### Cost Calculation

Computational cost for the numerical method is estimated as:

$$\text{Cost} = N_{\text{CG}} \times N^2 + N_{\text{matvec}} \times N^2$$

where:

- $N_{\text{CG}}$ = total CG iterations across all time steps
- $N_{\text{matvec}}$ = total sparse matrix-vector multiply operations
- Each CG iteration and matvec operation costs roughly $O(N^2)$

The cost depends on the tunable parameters:

- **N** (spatial resolution): affects $N^2$ term
- **dt** (time step): smaller dt → more time steps → more CG iterations
- **cg_atol** (CG tolerance): stricter tolerance → more CG iterations per solve

Note: The analytical solution (used only as reference for error checking) is not part of the optimization task.

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
   - Max iterations: 4 (N ∈ {32, 64, 128, 256})
   - **Non-target parameters**: dt∈{5.0, 10.0, 20.0}, cg_atol∈{1e0, 1e-1, 1e-2, 1e-3, 1e-4}

2. **dt Convergence Search (iterative+0-shot)**
   - For dummy solution, this means halving dt each iteration (multiplication factor: 0.5) starting from 40.0 until convergence
   - Max iterations: 4 (dt ∈ {40.0, 20.0, 10.0, 5.0})
   - **Non-target parameters**: N∈{64, 128, 256}, cg_atol∈{1e0, 1e-1, 1e-2, 1e-3, 1e-4}

3. **cg_atol Optimization (iterative+0-shot)**
   - For dummy solution, iteratively search from coarse (relaxed) to fine (strict) tolerance, stopping at the first convergent solution
   - Search follows a logarithmic progression through 5 predefined cg_atol values until convergence is achieved
   - **Non-target parameters**: N∈{64, 128, 256}, dt∈{5.0, 10.0, 20.0}

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| N | Grid resolution (number of grid points in each direction) | 32 ≤ N ≤ 256 |
| dt | Time step for RK4 integration | 5.0 ≤ dt ≤ 40.0 |
| cg_atol | Conjugate Gradient solver absolute tolerance | 1e-4 ≤ cg_atol ≤ 1e0 |

More Notes:

- **N (Grid Resolution)**: Determines spatial resolution via $\Delta x = \Delta y = L / N$ where $L \approx 62.83$
  - Smaller N → coarser grid (larger Δx) → lower accuracy but lower cost
  - Example: N=32 gives Δx≈1.96; N=256 gives Δx≈0.245
- **dt (Time Step)**: Determines temporal resolution over fixed total time T=10,000
  - Larger dt → fewer time steps ($N_{\text{steps}} = 10000/dt$) → lower accuracy but lower cost
  - Example: dt=5.0 needs 2,000 steps; dt=40.0 needs 250 steps
- **cg_atol (CG Tolerance)**: Controls CG solver convergence
  - Larger cg_atol → fewer CG iterations → lower accuracy but lower cost
  - cg_atol = 1e0 typically gives very inaccurate results due to loose CG convergence

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
| max_wall_time | Maximum computation time in seconds | 60.0 |
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

- **N** (iterative+0-shot): 4 profiles × 15 non-target combos (3 dt × 5 cg_atol) = 60 tasks
- **dt** (iterative+0-shot): 4 profiles × 15 non-target combos (3 N × 5 cg_atol) = 60 tasks
- **cg_atol** (iterative+0-shot): 4 profiles × 9 non-target combos (3 N × 3 dt) = 36 tasks
- **Total per precision**: 156 tasks
- **Total tasks**: 468 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/hasegawa_mima_linear.yaml`
Cache script: `checkouts/hasegawa_mima_linear.py`
