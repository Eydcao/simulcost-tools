# Hasegawa-Mima Linear Equation Solver

## Overview

The Hasegawa-Mima linear solver implements both analytical and numerical solutions for the linearized Hasegawa-Mima equation, which describes drift wave dynamics in plasma physics. This solver is part of the costsci-tools benchmark suite for computational cost analysis.

## Physics Background

The Hasegawa-Mima equation models drift wave turbulence in magnetized plasmas. The linear version solved here is:

```
∂q/∂t + v_star * ∂φ/∂y = 0
```

where:
- `q = ∇²φ - φ` is the generalized vorticity
- `φ` is the electrostatic potential
- `v_star` is the diamagnetic drift velocity
- The domain is periodic in both x and y directions

## Solver Methods

### 1. Numerical Method
- **Time integration**: 4th-order Runge-Kutta (RK4)
- **Spatial discretization**: Finite differences with periodic boundary conditions
- **Linear solver**: Conjugate Gradient (CG) for the Helmholtz equation `(∇² - I)φ = q`
- **Sparse matrices**: Uses scipy.sparse for efficient matrix operations

### 2. Analytical Method
- **Spectral solution**: Uses 2D FFT for exact solution in Fourier space
- **Time evolution**: Direct phase evolution `exp(i * v_star * ky * t / (1 + k²))`
- **Zero numerical error**: Provides reference solution for validation

## Parameters

### Environmental Parameters (Fixed)
- `L`: Domain size (2π × 10 ≈ 62.83)
- `v_star`: Diamagnetic drift velocity (0.02)
- `Dx`: Initial condition spatial scale (5.0)

### Tunable Parameters
- `N`: Grid resolution (main accuracy parameter, default: 256)
- `dt`: Time step (main efficiency parameter, default: 10.0)
- `cg_atol`: CG solver tolerance (default: 1e-8)
- `cg_maxiter`: CG solver maximum iterations (default: 1000)

### Simulation Control
- `analytical`: Method selection (true/false)
- `record_dt`: Time interval between recordings (default: 1000.0)
- `end_frame`: Number of recording frames (default: 10)

## Usage

### Basic Usage

```bash
# Analytical solution
python runners/hasegawa_mima_linear.py analytical=true

# Numerical solution
python runners/hasegawa_mima_linear.py analytical=false N=128 dt=10.0

# Parameter sweep
python runners/hasegawa_mima_linear.py N=64 dt=50.0 cg_atol=1e-4
```

### Python API

```python
from wrappers.hasegawa_mima_linear import run_sim_hasegawa_mima_linear

# Run numerical simulation
cost_numerical = run_sim_hasegawa_mima_linear("p1", N=128, dt=20.0, analytical=False)

# Run analytical simulation
cost_analytical = run_sim_hasegawa_mima_linear("p1", N=128, dt=1.0, analytical=True)

print(f"Numerical cost: {cost_numerical}")
print(f"Analytical cost: {cost_analytical}")
```

### Configuration Profiles

Default configurations are available in `run_configs/hasegawa_mima_linear/`:
- `p1.yaml`: Standard monopole configuration (Gaussian blob initial condition)
- `p2.yaml`: Dipole configuration (Gaussian dipole initial condition)
- `p3.yaml`: Sinusoidal configuration (Pure sinusoidal initial condition)
- `p4.yaml`: Mixed configuration (Sinusoidal in x, Gaussian in y)
- `p5.yaml`: Mixed configuration (Gaussian in x, sinusoidal in y)

Override parameters via command line:
```bash
python runners/hasegawa_mima_linear.py --config-name=p1 N=512 dt=5.0
```

## Output Files

The solver generates:

### HDF5 Files (`frame_XXXX.h5`)
- `phi`: Electrostatic potential field (2D array)
- `coordinates_x`, `coordinates_y`: Spatial coordinates
- Attributes: `time`, `N`, `dt`, `analytical`, `error`

### PNG Files (`frame_XXXX.png` and `frame_XXXX_spectrum.png`)
- `frame_XXXX.png`: Field visualization showing phi, analytical solution, and difference
- `frame_XXXX_spectrum.png`: Power spectrum analysis in k-space

### Metadata (`meta.json`)
- `cost`: Computational cost estimate (FLOPs)
- `error`: Mean L2 error vs analytical solution
- `n_steps`: Total simulation steps
- `cg_iterations_total`, `cg_calls`: CG solver statistics

## Cost Estimation

### Numerical Method
```
Cost = n_steps × 4 × (CG_cost + sparse_matvec_cost)
```
where:
- `n_steps`: Number of RK4 time steps
- `4`: RK4 requires 4 RHS evaluations per step
- `CG_cost`: Average CG iterations × N² operations per iteration
- `sparse_matvec_cost`: N² operations for sparse matrix-vector multiply

### Analytical Method
```
Cost = n_outputs × N² × log₂(N²)
```
where:
- `n_outputs`: Number of output times
- `N² × log₂(N²)`: Cost of 2D FFT operations

## Error Metrics

The solver computes L2 error against the analytical solution:
```
error = √(mean((φ_numerical - φ_analytical)²))
```

Typical error scaling:
- Higher resolution (larger N): Lower error
- Smaller time step (smaller dt): Lower error
- Tighter CG tolerance (smaller cg_atol): Lower error

## Performance Characteristics

### Computational Complexity
- **Numerical**: O(N² × n_steps × CG_iterations)
- **Analytical**: O(N² × log(N²) × n_outputs)

### Memory Usage
- **Sparse matrices**: O(N²) storage for Laplacian and derivative operators
- **State vectors**: O(N²) for solution fields
- **FFT workspace**: O(N²) for analytical method

### Scalability
- Grid resolution N: Quadratic scaling in memory and operations
- Time steps: Linear scaling in operations
- CG iterations: Depends on condition number and tolerance

## Validation

The solver includes built-in validation:
1. **Analytical vs Numerical**: Compare numerical solution with analytical reference
2. **Conservation**: Monitor energy/enstrophy conservation properties
3. **Convergence**: CG solver convergence monitoring

## Example Results

Typical parameter sets and their characteristics:

| Configuration | N   | dt   | Error   | Cost (numerical) | Cost (analytical) |
|---------------|-----|------|---------|------------------|-------------------|
| Fast          | 64  | 50.0 | ~1e-3   | ~1e7            | ~1e5             |
| Balanced      | 128 | 20.0 | ~1e-4   | ~1e8            | ~4e5             |
| Accurate      | 256 | 10.0 | ~1e-5   | ~1e9            | ~2e6             |

## Troubleshooting

### Common Issues

1. **CG Solver Convergence**
   - Increase `cg_maxiter` or decrease `cg_atol`
   - Check for numerical instabilities

2. **High Error**
   - Decrease time step `dt`
   - Increase resolution `N`
   - Tighten CG tolerance `cg_atol`

3. **Memory Issues**
   - Reduce grid resolution `N`
   - Use analytical method for large N

### Performance Optimization

1. **For Accuracy**: Decrease `dt`, increase `N`, tighten `cg_atol`
2. **For Speed**: Increase `dt`, decrease `N`, relax `cg_atol`
3. **For Large Grids**: Use analytical method when possible

## LLM Parameter Search Task Configuration

### Checkout Procedure

This solver provides an automated parameter optimization task generation system for LLM-based parameter search. The checkout process generates dummy solution datasets for training and evaluation.

### Task Distribution Strategy

Following the flexible parameter strategy, the solver generates approximately **225 individual tasks** distributed across:

- **5 profiles** (p1: monopole, p2: dipole, p3: sinusoidal, p4: sin_x_gauss_y, p5: gauss_x_sin_y)
- **3 precision levels** (low, medium, high tolerance)
- **3 target parameters** (N, dt, cg_atol)
- **Multiple non-target parameter combinations**

### Target Parameters and Search Types

#### N (Grid Resolution) - Iterative + 0-shot Search
- **Description**: Spatial discretization resolution determining accuracy
- **Search Method**: Iterative refinement starting from N=64 + 0-shot exploration
- **Multiplication Factor**: 2 (doubles resolution each iteration)
- **Max Iterations**: 3 (limits to N: 64, 128, 256)
- **Non-target Parameters**:
  - dt: [5.0, 10.0, 20.0]
  - cg_atol: [1e-4, 1e-5, 1e-6]

#### dt (Time Step) - Iterative + 0-shot Search
- **Description**: Temporal discretization controlling numerical stability
- **Search Method**: Iterative refinement starting from dt=10.0 + 0-shot exploration
- **Multiplication Factor**: 0.5 (halves time step each iteration)
- **Max Iterations**: 3 (limits to dt: 10.0, 5.0, 2.5)
- **Non-target Parameters**:
  - N: 128 (fixed moderate resolution)
  - cg_atol: [1e-4, 1e-5, 1e-6]

#### cg_atol (CG Tolerance) - 0-shot Search
- **Description**: Conjugate gradient solver convergence tolerance
- **Search Method**: Grid search over logarithmic range [1e-8, 1e-3]
- **Search Points**: 4 logarithmically spaced values
- **Non-target Parameters**:
  - N: 128 (fixed moderate resolution)
  - dt: [5.0, 10.0, 20.0]

### Precision Levels

Convergence tolerances based on error vs analytical solution:

```yaml
precision_levels:
  low:
    tolerance_rmse: 0.001      # Relaxed convergence
  medium:
    tolerance_rmse: 0.0005     # Moderate convergence
  high:
    tolerance_rmse: 0.0002     # Stringent convergence
```

### Task Breakdown

**Task distribution per precision level:**
- N parameter: 5 profiles × 9 combinations (3 dt × 3 cg_atol) = 45 tasks
- dt parameter: 5 profiles × 3 combinations (1 N × 3 cg_atol) = 15 tasks
- cg_atol parameter: 5 profiles × 3 combinations (1 N × 3 dt) = 15 tasks
- **Total per precision**: 75 tasks
- **Total tasks**: 225 tasks (across 3 precision levels)

### Profile Configurations

#### p1 (Monopole)
- Standard monopole (Gaussian blob) initial condition
- Dx=5.0, 10 recording frames over 10,000 time units
- Tests convergence for smooth, localized initial conditions

#### p2 (Dipole)
- Dipole (Gaussian dipole) initial condition
- Dx=5.0, 10 recording frames over 10,000 time units
- Tests convergence for antisymmetric initial conditions

#### p3 (Sinusoidal)
- Pure sinusoidal initial condition
- Dx=5.0, 10 recording frames over 10,000 time units
- Tests convergence for periodic initial conditions

#### p4 (Mixed: Sin x, Gauss y)
- Sinusoidal in x, Gaussian in y initial condition
- Dx=5.0, 10 recording frames over 10,000 time units
- Tests convergence for partially periodic initial conditions

#### p5 (Mixed: Gauss x, Sin y)
- Gaussian in x, sinusoidal in y initial condition
- Dx=5.0, 10 recording frames over 10,000 time units
- Tests convergence for partially periodic initial conditions

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/hasegawa_mima_linear.yaml`

Cache script: `checkouts/hasegawa_mima_linear.py`

### Running Checkout Generation

```bash
# Generate all dummy solutions for LLM training
cd /path/to/costsci-tools
python checkouts/hasegawa_mima_linear.py

# This will create:
# - dataset/hasegawa_mima_linear/successful/tasks.json
# - dataset/hasegawa_mima_linear/failed/tasks.json
# - outputs/statistics/hasegawa_mima_linear_statistics.png
# - outputs/statistics/hasegawa_mima_linear_statistics_summary.txt
```

The generated datasets contain complete parameter optimization trajectories with:
- Initial parameters and search configurations
- Convergence results and optimal parameter values
- Cost histories and computational trajectories
- Success/failure classifications for LLM training

## References

1. Hasegawa, A. & Mima, K. (1978). "Pseudo-three-dimensional turbulence in magnetized nonuniform plasma." Physics of Fluids, 21(1), 87-92.
2. Scott, B. (2002). "The nonlinear drift wave instability and its role in tokamak edge turbulence." New Journal of Physics, 4(1), 52.