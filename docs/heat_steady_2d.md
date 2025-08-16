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
2. **p2** - Opposite hot sides: Top and bottom hot (T=1.0), left and right cold (T=0.0)
3. **p3** - Corner heating: Right side hot (T=1.0), others cold (T=0.0)
4. **p4** - Mixed pattern: Left hot (T=1.0), top/bottom moderate (T=0.5), right cold (T=0.0)
5. **p5** - Uniform heating: All boundaries warm (T=0.8)

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
   - For dummy solution, halve dx each round (multiplication factor: 0.5) starting from 0.01 until convergence
   - **Non-target parameters**: relax=1.0, error_threshold=1e-8, T_init=0.0

2. **relax Optimization (0-shot)**
   - For dummy solution, grid search the relax that achieves convergence with minimum computational cost
   - **Non-target parameters**: dx=0.01, error_threshold=1e-8, T_init=0.0

3. **T_init Optimization (0-shot)**
   - For dummy solution, grid search the T_init that achieves convergence with minimum computational cost
   - **Non-target parameters**: dx=0.01, relax=1.0, error_threshold=1e-8

4. **error_threshold Convergence Search (iterative+0-shot)**
   - For dummy solution, reduce error_threshold each round (multiplication factor: 0.1) starting from 1e-5 until convergence
   - **Non-target parameters**: dx=0.01, relax=1.0, T_init=0.0

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
  - **p2**: Opposite heating (top and bottom heated)
  - **p3**: Corner heating (side boundary heated)
  - **p4**: Mixed boundary conditions (complex gradient patterns)
  - **p5**: Uniform heating (convergence to steady state)
- **Target Parameters**: 4 (dx, relax, error_threshold, T_init)
- **Precision Levels**: 3 (low: 1e-4, medium: 1e-5, high: 1e-6)

### Task Distribution

Current configuration generates:

- **dx** (iterative+0-shot): 5 profiles × 4 non-target combos = 20 tasks
- **relax** (0-shot): 5 profiles × 3 non-target combos = 15 tasks
- **T_init** (0-shot): 5 profiles × 3 non-target combos = 15 tasks
- **error_threshold** (iterative+0-shot): 5 profiles × 3 non-target combos = 15 tasks
- **Total per precision**: 65 tasks
- **Total tasks**: 195 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/heat_steady_2d.yaml`
Cache script: `checkouts/heat_steady_2d.py`

## Checkout

### Summary

- **Benchmarks**:
  - **p1**: Classic one-hot-side problem (top boundary heated)
  - **p2**: Opposite heating (top and bottom heated)  
  - **p3**: Corner heating (side boundary heated)
  - **p4**: Mixed boundary conditions (complex gradient patterns)
  - **p5**: Uniform heating (convergence to steady state)
- **Target Parameters**: 4 (dx, relax, error_threshold, T_init)
- **Precision Levels**: 3 (low: 1e-4, medium: 1e-5, high: 1e-6)

### Task Distribution

Current configuration generates:

- **dx** (iterative+0-shot): 5 profiles × 4 non-target combos = 20 tasks
- **error_threshold** (iterative+0-shot): 5 profiles × 3 non-target combos = 15 tasks
- **relax** (0-shot): 5 profiles × 3 non-target combos = 15 tasks  
- **T_init** (0-shot): 5 profiles × 3 non-target combos = 15 tasks
- **Total per precision**: 65 tasks
- **Total tasks**: 195 tasks (across 3 precision levels)

### Dummy Solution Strategy

1. **dx (iterative+0-shot)**:
   - Start with dx=0.01, halve each iteration until spatial convergence
   - Non-target: relax=1.0, error_threshold=1e-8, T_init varies

2. **error_threshold (iterative+0-shot)**:
   - Start with error_threshold=1e-5, reduce by factor 0.1 until convergence
   - Non-target: dx=0.01, relax=1.0, T_init varies

3. **relax (0-shot)**:
   - Grid search over [0.1, 1.9] with 10 values to find minimum cost
   - Non-target: dx=0.01, error_threshold=1e-8, T_init varies

4. **T_init (0-shot)**:
   - Grid search over [0.0, 1.0] with 10 values to find minimum cost
   - Non-target: dx=0.01, relax=1.0, error_threshold varies

### Running Checkout Generation

```bash
# Generate all dummy solution cache
python checkouts/heat_steady_2d.py

# Output locations:
# - Dataset: dataset/heat_steady_2d/successful/tasks.json
#           dataset/heat_steady_2d/failed/tasks.json  
# - Statistics: outputs/statistics/heat_steady_2d_statistics.png
#              outputs/statistics/heat_steady_2d_statistics_summary.txt
```
