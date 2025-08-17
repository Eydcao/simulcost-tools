# Steady State Heat Transfer in 2D with SOR Method

## Introduction

This simulation solves 2D steady-state heat transfer problems using the Jacobi iteration method with Successive Over-Relaxation (SOR). The solver handles rectangular domains with fixed boundary conditions using an iterative approach.

**Governing equation:**
$$\nabla^2 T = 0$$

**Discretized form (5-point stencil):**
$$T_{i,j} = \frac{1}{4}(T_{i-1,j} + T_{i+1,j} + T_{i,j-1} + T_{i,j+1})$$

**SOR update formula:**
$$T_{i,j}^{new} = \omega \cdot T_{i,j}^{Jacobi} + (1-\omega) \cdot T_{i,j}^{old}$$

where $\omega$ is the relaxation parameter.

### Numerical Method

The solution uses point-wise Jacobi iteration with SOR acceleration:

1. **Jacobi Update**: Calculate new temperature based on neighboring values
2. **SOR Relaxation**: Blend new and old values using relaxation parameter $\omega$
3. **Convergence Check**: Monitor RMSE between successive iterations
4. **Boundary Enforcement**: Maintain fixed boundary conditions at each iteration

### Boundary Conditions

The solver supports arbitrary fixed temperature boundary conditions:

- **Top boundary**: $T(x, L_y) = T_{top}$
- **Bottom boundary**: $T(x, 0) = T_{bottom}$
- **Left boundary**: $T(0, y) = T_{left}$
- **Right boundary**: $T(L_x, y) = T_{right}$

Corner temperatures are set as the average of adjacent boundary values.

## Test Cases

The profile configurations define different boundary condition patterns:

1. **p1** - Classic case: Top hot (T=1.0), others cold (T=0.0)
2. **p2-p8**: Random boundary conditions

The simulated results are considered correct if the relative RMSE meets the precision-dependent tolerance and the solution satisfies physical constraints:

1. **Temperature validity**: All values finite and within boundary range
2. **Gradient reasonableness**: Temperature gradients remain physically reasonable

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **dx Convergence Search (iterative+0-shot)**
   - Grid spacing determines spatial resolution: $\Delta x = L_x / n_x$, $\Delta y = L_y / n_y$

2. **relax Optimization (0-shot)**
   - Grid search over $\omega \in [0.1, 1.9]$ to find optimal SOR relaxation parameter

3. **T_init Optimization (0-shot)**
   - Grid search over initial temperature field values to minimize convergence time

4. **error_threshold Convergence Search (iterative+0-shot)**
   - Convergence threshold for stopping iteration when RMSE between steps drops below threshold

### Dummy Strategy

1. **dx Convergence Search (iterative+0-shot)**
   - For dummy solution, halve dx each round (multiplication factor: 0.5) starting from 0.08 until convergence
   - **Non-target parameters**: relax=[0.2,0.6,1.0], error_threshold=1e-8, t_init=[0.0,0.25,0.5,0.75,1.0]

2. **relax Optimization (0-shot)**
   - For dummy solution, grid search the relax that achieves convergence with minimum computational cost
   - **Non-target parameters**: dx=0.01, error_threshold=1e-8, t_init=[0.0,0.25,0.5,0.75,1.0]

3. **t_init Optimization (0-shot)**
   - For dummy solution, grid search the t_init that achieves convergence with minimum computational cost
   - **Non-target parameters**: dx=0.01, relax=[0.2,0.6,1.0], error_threshold=1e-8

4. **error_threshold Convergence Search (iterative+0-shot)**
   - For dummy solution, reduce error_threshold each round (multiplication factor: 0.1) starting from 1e-4 until convergence
   - **Non-target parameters**: dx=0.01, relax=[0.2,0.6,1.0], t_init=[0.0,0.25,0.5,0.75,1.0]

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| dx | Grid spacing (determines resolution) | 0.001 ≤ dx ≤ 0.1 |
| relax | SOR relaxation parameter | 0 < relax < 2 |
| error_threshold | Convergence criterion (RMSE between iterations) | 1e-12 ≤ error_threshold ≤ 1e-4 |
| T_init | Initial temperature field value | Domain-dependent |

More Notes:

- $\omega = 1$: Standard Jacobi iteration (no acceleration)
- $\omega < 1$: Under-relaxation (more stable, slower convergence)
- $\omega > 1$: Over-relaxation (faster convergence if stable)
- $\omega$ must be in (0, 2) for convergence
- Optimal $\omega$ depends on grid size and boundary conditions
- $dx$ determines spatial resolution: smaller $dx$ = finer grid = higher accuracy but higher cost

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| Lx | Domain length in x | 1.0 |
| Ly | Domain length in y | 1.0 |
| T_top | Top boundary temperature | Profile-dependent |
| T_bottom | Bottom boundary temperature | Profile-dependent |
| T_left | Left boundary temperature | Profile-dependent |
| T_right | Right boundary temperature | Profile-dependent |
| record_dt | Iteration interval between recordings | 100 |
| end_frame | Simulation end after certain number of frames | 50 |
| dump_dir | Directory for output files | "sim_res/heat_steady_2d/p1" |
| verbose | Enable verbose output | False |

## Checkout

### Summary

- **Benchmarks**:
  - **p1**: Classic one-hot-side problem (top boundary heated)
  - **p2-p8**: Random boundary conditions
- **Target Parameters**: 4 (dx, relax, error_threshold, t_init)
- **Precision Levels**: 3 (low: 0.05, medium: 0.005, high: 0.0005)

### Task Distribution

Current configuration generates:

- **dx** (iterative+0-shot): 8 profiles × 15 non-target combos = 120 tasks
- **relax** (0-shot): 8 profiles × 5 non-target combos = 40 tasks
- **t_init** (0-shot): 8 profiles × 3 non-target combos = 24 tasks
- **error_threshold** (iterative+0-shot): 8 profiles × 15 non-target combos = 120 tasks
- **Total per precision**: 304 tasks
- **Total tasks**: 912 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/heat_steady_2d.yaml`
Cache script: `checkouts/heat_steady_2d.py`
