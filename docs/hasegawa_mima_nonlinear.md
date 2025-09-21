# Hasegawa-Mima Nonlinear Equation Solver

## Overview

The Hasegawa-Mima nonlinear solver implements a pseudo-spectral numerical solution for the nonlinear Hasegawa-Mima equation, which describes drift wave turbulence in magnetized plasmas. This solver is part of the costsci-tools benchmark suite for computational cost analysis.

## Physics Background

The Hasegawa-Mima equation models drift wave turbulence in magnetized plasmas. The nonlinear version solved here is:

```
∂q/∂t + {φ, q} + v_star * ∂φ/∂y = 0
```

where:
- `q = ∇²φ - φ` is the generalized vorticity
- `φ` is the electrostatic potential
- `{φ, q}` is the Poisson bracket (Jacobian): `∂φ/∂x * ∂q/∂y - ∂φ/∂y * ∂q/∂x`
- `v_star` is the diamagnetic drift velocity
- The domain is periodic in both x and y directions

## Solver Method

### Pseudo-Spectral Approach
- **Spatial discretization**: 2D FFT for computing derivatives
- **Time integration**: 4th-order Runge-Kutta (RK4)
- **Nonlinear terms**: Computed in physical space with dealiasing
- **Dealiasing**: 2/3 rule to prevent aliasing errors
- **Linear terms**: Computed in spectral space

### Algorithm Details
1. **Initialization**: Transform initial condition to spectral space
2. **Time stepping**: RK4 integration with spectral RHS evaluation
3. **RHS computation**:
   - Solve Poisson equation: `φ = q / (∇² - I)` in spectral space
   - Compute Poisson bracket `{φ, q}` with dealiasing in physical space
   - Add linear drift term `v_star * ∂φ/∂y` in spectral space
4. **Output**: Transform back to physical space for visualization

## Parameters

### Environmental Parameters (Fixed)
- `L`: Domain size (2π × 10 ≈ 62.83)
- `v_star`: Diamagnetic drift velocity (0.02)
- `Dx`: Initial condition spatial scale (5.0)
- `dealias_ratio`: Dealiasing ratio (2/3)

### Tunable Parameters
- `N`: Grid resolution (main accuracy parameter, default: 128)
- `dt`: Time step (main efficiency parameter, default: 10.0)

### Simulation Control
- `record_dt`: Time interval between recordings (default: 1000.0)
- `end_frame`: Number of recording frames (default: 10)

## Usage

### Basic Usage

```bash
# Standard nonlinear simulation
python runners/hasegawa_mima_nonlinear.py

# With custom parameters
python runners/hasegawa_mima_nonlinear.py N=256 dt=5.0

# Different initial condition
python runners/hasegawa_mima_nonlinear.py --config-name=p2 N=128 dt=10.0
```

### Python API

```python
from wrappers.hasegawa_mima_nonlinear import run_sim_hasegawa_mima_nonlinear

# Run nonlinear simulation
cost = run_sim_hasegawa_mima_nonlinear("p1", N=128, dt=10.0)

print(f"Nonlinear simulation cost: {cost}")
```

### Configuration Profiles

Default configurations are available in `run_configs/hasegawa_mima_nonlinear/`:
- `p1.yaml`: Standard monopole configuration (Gaussian blob initial condition)
- `p2.yaml`: Dipole configuration (Gaussian dipole initial condition)
- `p3.yaml`: Sinusoidal configuration (Pure sinusoidal initial condition)
- `p4.yaml`: Mixed configuration (Sinusoidal in x, Gaussian in y)
- `p5.yaml`: Mixed configuration (Gaussian in x, sinusoidal in y)

Override parameters via command line:
```bash
python runners/hasegawa_mima_nonlinear.py --config-name=p1 N=256 dt=5.0
```

## Output Files

The solver generates:

### HDF5 Files (`frame_XXXX.h5`)
- `phi`: Electrostatic potential field (2D array)
- `coordinates_x`, `coordinates_y`: Spatial coordinates
- Attributes: `time`, `N`, `dt`, `dealias_ratio`, `fft_operations`, `poisson_bracket_calls`

### PNG Files (`frame_XXXX.png`)
- Field visualization showing phi evolution over time

### Metadata (`meta.json`)
- `cost`: Computational cost estimate (FLOPs)
- `n_steps`: Total simulation steps
- `fft_operations`: Total FFT operations performed
- `poisson_bracket_calls`: Number of nonlinear term evaluations

## Cost Estimation

### Computational Cost
```
Cost = fft_operations × N² × log₂(N²)
```
where:
- `fft_operations`: Total number of FFT/IFFT operations
- Dominated by FFT operations in RHS evaluation
- Each RHS call requires ~9 FFTs (derivatives + Poisson bracket + final transform)

### FFT Operation Breakdown
Per RHS evaluation:
- 4 FFTs for computing derivatives of φ and q
- 4 IFFTs for transforming to physical space
- 1 FFT for transforming Poisson bracket back to spectral space

## Convergence Validation

Since there's no analytical solution for the nonlinear case, validation uses traditional resolution convergence:

### Resolution Convergence
1. **Method**: Compare solutions at different resolutions (N, 2N)
2. **Metric**: L2 error between downsampled fine and coarse solutions
3. **Criterion**: Error below tolerance indicates convergence

### Convergence Check
```python
from wrappers.hasegawa_mima_nonlinear import compare_resolutions

# Compare N=128 vs N=256 solutions
comparison = compare_resolutions(
    "sim_res/hasegawa_mima_nonlinear/p1_N_128_dt_1.00e+01_nonlinear",
    "sim_res/hasegawa_mima_nonlinear/p1_N_256_dt_1.00e+01_nonlinear"
)

print(f"L2 error: {comparison['mean_l2_error']:.2e}")
```

## Performance Characteristics

### Computational Complexity
- **Spatial operations**: O(N² × log(N²)) per time step
- **Time steps**: Linear scaling with simulation time
- **Nonlinear terms**: Additional O(N²) operations per step

### Memory Usage
- **Spectral fields**: O(N²) complex arrays
- **Physical space fields**: O(N²) real arrays
- **FFT workspace**: O(N²) for transform operations

### Scalability
- Grid resolution N: O(N² × log(N²)) scaling
- Time steps: Linear scaling
- Dealiasing overhead: ~3x increase in operations vs no dealiasing

## Validation

The solver includes built-in validation:
1. **Resolution Convergence**: Compare different grid resolutions
2. **Energy Conservation**: Monitor energy conservation properties
3. **Stability**: Check for numerical instabilities

## Example Results

Typical parameter sets and their characteristics:

| Configuration | N   | dt   | Cost (approx) | Convergence |
|---------------|-----|------|---------------|-------------|
| Fast          | 64  | 20.0 | ~1e7         | Moderate    |
| Balanced      | 128 | 10.0 | ~4e7         | Good        |
| Accurate      | 256 | 5.0  | ~2e8         | Excellent   |

## Troubleshooting

### Common Issues

1. **Numerical Instabilities**
   - Decrease time step `dt`
   - Increase resolution `N`
   - Check for negative time steps

2. **Poor Convergence**
   - Increase resolution `N`
   - Decrease time step `dt`
   - Verify dealiasing is working

3. **Memory Issues**
   - Reduce grid resolution `N`
   - Monitor memory usage during FFTs

### Performance Optimization

1. **For Accuracy**: Increase `N`, decrease `dt`
2. **For Speed**: Decrease `N`, increase `dt` (within stability limits)
3. **For Large Grids**: Consider parallel FFT implementations

## Implementation Status

**⚠️ PROOF-OF-CONCEPT IMPLEMENTATION**

This is a proof-of-concept implementation of the nonlinear Hasegawa-Mima solver. The core solver is functional and can run simulations, but the full parameter optimization and checkout system requires optimization for practical use.

### Current Status

✅ **Implemented and Working:**
- Pseudo-spectral nonlinear solver with RK4 time integration
- 2/3 rule dealiasing for nonlinear terms
- All 5 initial condition profiles (monopole, dipole, sinusoidal, mixed)
- Basic wrapper and runner infrastructure
- Simple test cases (N=32, N=64)

⚠️ **Limitations:**
- Simulations are computationally expensive for full parameter optimization
- Convergence checking between resolutions needs optimization
- Full checkout may require substantial computational time

### Simple Test Results

The basic implementation successfully runs:
```bash
python checkouts/hasegawa_mima_nonlinear_simple.py
```

Test results show:
- N=32, dt=10.0: Cost ≈ 3.7e6 FLOPs
- N=64, dt=10.0: Cost ≈ 5.3e7 FLOPs

## LLM Parameter Search Task Configuration (Future Work)

### Planned Task Distribution Strategy

When fully optimized, the solver would generate approximately **90 individual tasks** distributed across:

- **5 profiles** (p1: monopole, p2: dipole, p3: sinusoidal, p4: sin_x_gauss_y, p5: gauss_x_sin_y)
- **3 precision levels** (low, medium, high tolerance)
- **2 target parameters** (N, dt)
- **Multiple non-target parameter combinations**

### Target Parameters and Search Types

#### N (Grid Resolution) - Iterative Search
- **Description**: Spatial discretization resolution determining accuracy
- **Search Method**: Iterative refinement starting from N=64
- **Multiplication Factor**: 2 (doubles resolution each iteration)
- **Max Iterations**: 3 (limits to N: 64, 128, 256)
- **Non-target Parameters**:
  - dt: [5.0, 10.0, 20.0]

#### dt (Time Step) - Iterative Search
- **Description**: Temporal discretization controlling numerical stability
- **Search Method**: Iterative refinement starting from dt=10.0
- **Multiplication Factor**: 0.5 (halves time step each iteration)
- **Max Iterations**: 3 (limits to dt: 10.0, 5.0, 2.5)
- **Non-target Parameters**:
  - N: [128] (fixed moderate resolution)

### Precision Levels

Convergence tolerances based on L2 error between resolutions:

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
- N parameter: 5 profiles × 3 combinations (3 dt values) = 15 tasks
- dt parameter: 5 profiles × 1 combination (1 N value) = 5 tasks
- **Total per precision**: 18 tasks
- **Total tasks**: 90 tasks (across 3 precision levels × 5 profiles)

### Profile Configurations

#### p1 (Monopole)
- Standard monopole (Gaussian blob) initial condition
- Tests convergence for smooth, localized initial conditions

#### p2 (Dipole)
- Dipole (Gaussian dipole) initial condition
- Tests convergence for antisymmetric initial conditions

#### p3 (Sinusoidal)
- Pure sinusoidal initial condition
- Tests convergence for periodic initial conditions

#### p4 (Mixed: Sin x, Gauss y)
- Sinusoidal in x, Gaussian in y initial condition
- Tests convergence for partially periodic initial conditions

#### p5 (Mixed: Gauss x, Sin y)
- Gaussian in x, sinusoidal in y initial condition
- Tests convergence for partially periodic initial conditions

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/hasegawa_mima_nonlinear.yaml`

Cache script: `checkouts/hasegawa_mima_nonlinear.py`

### Running Simple Test

```bash
# Run basic functionality test
cd /path/to/costsci-tools
python checkouts/hasegawa_mima_nonlinear_simple.py

# This creates:
# - dataset/hasegawa_mima_nonlinear/test/results.json
```

### Future Full Checkout (When Optimized)

```bash
# Generate all dummy solutions for LLM training (future implementation)
cd /path/to/costsci-tools
python checkouts/hasegawa_mima_nonlinear.py

# Would create:
# - dataset/hasegawa_mima_nonlinear/successful/tasks.json
# - dataset/hasegawa_mima_nonlinear/failed/tasks.json
# - outputs/statistics/hasegawa_mima_nonlinear_statistics.png
# - outputs/statistics/hasegawa_mima_nonlinear_statistics_summary.txt
```

The generated datasets would contain complete parameter optimization trajectories with:
- Initial parameters and search configurations
- Convergence results and optimal parameter values
- Cost histories and computational trajectories
- Success/failure classifications for LLM training

## Implementation Notes

### Performance Optimization Needed

For practical use, the following optimizations are recommended:

1. **Reduced Simulation Time**: Use shorter end times for parameter optimization
2. **Coarser Initial Resolution**: Start with N=32 or N=64 for faster iterations
3. **Parallel Processing**: Implement parallel checkout processing
4. **Adaptive Tolerances**: Use relaxed convergence criteria for initial searches

### Current Configuration

The current implementation uses reduced parameters for testing:
- `record_dt: 50` (vs 1000 in linear solver)
- `end_frame: 2` (vs 10 in linear solver)
- `Initial amplitude: 0.1` (for numerical stability)

## References

1. Hasegawa, A. & Mima, K. (1978). "Pseudo-three-dimensional turbulence in magnetized nonuniform plasma." Physics of Fluids, 21(1), 87-92.
2. Scott, B. (2002). "The nonlinear drift wave instability and its role in tokamak edge turbulence." New Journal of Physics, 4(1), 52.
3. Canuto, C. et al. (2006). "Spectral Methods: Fundamentals in Single Domains." Springer-Verlag.
4. Boyd, J. P. (2001). "Chebyshev and Fourier Spectral Methods." Dover Publications.