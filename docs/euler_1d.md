# Euler 1D Equations with 2nd Order MUSCL-Roe Method

## Introduction

This simulation solves the 1D Euler equations for compressible inviscid flow, using a 2nd order MUSCL scheme with Roe flux and generalized superbee limiter:

**Conservative form:**
$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = 0$$

Where the conservative variables and flux are:
$$\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho E \end{pmatrix}, \quad \mathbf{F} = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ u(\rho E + p) \end{pmatrix}$$

**Primitive variables:**

- $\rho$ = density
- $u$ = velocity  
- $p$ = pressure
- $E$ = specific total energy

**Equation of state:**
$$p = (\gamma - 1) \rho \left(E - \frac{u^2}{2}\right)$$

where $\gamma$ is the ratio of specific heats.

### Spatial Discretization

The spatial discretization uses MUSCL reconstruction with blending parameter $k$:

$$\mathbf{U}^L_{j+\frac{1}{2}} = \mathbf{U}_j + \frac{1+k}{4} \psi(r_{j}) (\mathbf{U}_{j+1} - \mathbf{U}_{j})$$

$$\mathbf{U}^R_{j+\frac{1}{2}} = \mathbf{U}_{j+1} - \frac{1+k}{4} \psi(r_{j+1}) (\mathbf{U}_{j+2} - \mathbf{U}_{j+1})$$

where $k$ is a blending coefficient between central ($k=1$) and upwind ($k=-1$) scheme, and $\psi(r)$ is the slope limiter function.

### Slope Limiting

The slope limiter uses a generalized superbee limiter:

$$\psi(r) = \max\left[0, \max\left[\min(\beta r, 1), \min(r, \beta)\right]\right]$$

where $\beta$ is the limiter parameter controlling dissipation.

The slope ratio $r$ at interface $j$ is defined as:

$$r_{j} = \frac{\mathbf{U}_{j+1} - \mathbf{U}_{j}}{\mathbf{U}_{j+2} - \mathbf{U}_{j+1}}$$

This ratio indicates the local non-smoothness, which will be the input into the slope limiter to achieve the TVD condition.

### Flux Computation

The interface flux is computed using the Roe approximate Riemann solver:

$$\mathbf{F}_{j+\frac{1}{2}} = \frac{1}{2}\left[\mathbf{F}(\mathbf{U}^L) + \mathbf{F}(\mathbf{U}^R)\right] - \frac{1}{2}|\mathbf{A}|(\mathbf{U}^R - \mathbf{U}^L)$$

where $|\mathbf{A}|$ is the Roe matrix with Roe-averaged quantities.

## Test Cases

The case key in the config file solver sets different initial conditions:

1. **sod** - Sod's shock tube problem:
   - Left: $\rho=1.0, u=0.0, p=1.0$
   - Right: $\rho=0.125, u=0.0, p=0.1$

2. **lax** - Lax problem:
   - Left: $\rho=0.445, u=0.6977, p=3.528$
   - Right: $\rho=0.5, u=0.0, p=0.571$

3. **mach_3** - Mach 3 problem:
   - Left: $\rho=3.857, u=0.92, p=10.333$
   - Right: $\rho=1.0, u=3.55, p=1.0$

The simulated results are considered correct if the relative RMSE meets the precision-dependent tolerance (low: 0.01, medium: 0.005, high: 0.0025) compared to reference solution, and the solution satisfies the convergence criteria:

1. **Positivity preservation**: pressure and density must remain positive at all times
2. **Shock speed consistency**: pressure gradients should not exceed physical bounds

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **CFL Convergence Search (iterative+0-shot)**
   - CFL (Courant-Friedrichs-Lewy) number is defined as: $CFL = \frac{(|u| + c) \Delta t}{\Delta x}$ where $c = \sqrt{\gamma p/\rho}$ is the speed of sound

2. **n_space Convergence Search (iterative+0-shot)**
   - n_space determines spatial resolution: $\Delta x = L / n\_space$, where $L$ is domain length

3. **β-Parameter Optimization (0-shot)**
   - Grid search over β ∈ [1.0, 2.0] with 6 equally spaced values to find the optimal limiter parameter

4. **k-Parameter Optimization (0-shot)**
   - Grid search over k ∈ [-1.0, 1.0] with 11 equally spaced values to find the optimal blending parameter

### Dummy Strategy

1. **CFL Convergence Search (iterative+0-shot)**
   - For dummy solution, this means halve CFL each round (multiplication factor: 0.5) starting from 1.0 until convergence
   - **Non-target parameters**: n_space=256, with β∈{1.0, 1.5, 2.0} and k∈{-1.0, 0.0, 1.0} combinations

2. **n_space Convergence Search (iterative+0-shot)**
   - For dummy solution, this means doubling n_space each iteration (multiplication factor: 2) starting from 256 until convergence
   - **Non-target parameters**: CFL=0.25, with β∈{1.0, 1.5, 2.0} and k∈{-1.0, 0.0, 1.0} combinations

3. **β-Parameter Optimization (0-shot)**
   - For dummy solution, grid search the β that achieves convergence with minimum computational cost
   - **Non-target parameters**: CFL=0.25, with k∈{-1.0, 0.0, 1.0} combinations, n_space=256 (starting)

4. **k-Parameter Optimization (0-shot)**
   - For dummy solution, grid search the k that achieves convergence with minimum computational cost
   - **Non-target parameters**: CFL=0.25, with β∈{1.0, 1.5, 2.0} combinations, n_space=256 (starting)

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| cfl | Courant-Friedrichs-Lewy number for stability | 0 < cfl ≤ 1 |
| beta | Limiter parameter for generalized superbee | 1 ≤ beta ≤ 2 |
| k | Blending parameter between central (k=1) and upwind (k=-1) fluxes | -1 ≤ k ≤ 1 |
| n_space | Number of grid cells for spatial discretization | 64 ≤ n_space ≤ 2048 |

More Notes:

- $\beta = 1$: minmod limiter (most dissipative)
- $\beta = 2$: superbee limiter (least dissipative)
- $\beta$ must not be smaller than 1 otherwise symmetry will be broken
- When $k = -1$, $\beta$ no longer affects the solution
- $n\_space$ determines spatial resolution: $\Delta x = L / n\_space$ (smaller $\Delta x$ = finer grid = higher accuracy but higher cost)

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| L | Domain length | 1.0 |
| gamma | Ratio of specific heats | 1.4 |
| case | Initial condition type | "sod" |
| record_dt | Time interval between recordings | 0.02 |
| end_frame | Simulation end after certain number of frames | 10 |
| dump_dir | Directory for output files | "sim_res/euler_1d/p1" |
| verbose | Enable verbose output | False |

## Checkout

### Summary

- **Benchmarks**:
  - **p1**: Sod shock tube problem (classic Riemann problem)
  - **p2**: Lax problem (moderate shock strength)
  - **p3**: Mach 3 problem (high-speed flow)
- **Target Parameters**: 4 (CFL, n_space, k, beta)
- **Precision Levels**: 3 (low: 0.01, medium: 0.005, high: 0.0025)

### Task Distribution

Current configuration generates:

- **CFL** (iterative+0-shot): 3 profiles × 9 non-target combos = 27 tasks
- **n_space** (iterative+0-shot): 3 profiles × 9 non-target combos = 27 tasks  
- **k** (0-shot): 3 profiles × 3 non-target combos = 9 tasks
- **beta** (0-shot): 3 profiles × 3 non-target combos = 9 tasks
- **Total per precision**: 72 tasks
- **Total tasks**: 216 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/euler_1d.yaml`
Cache script: `checkouts/euler_1d.py`
