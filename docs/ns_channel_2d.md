# Navier-Stokes Channel 2D Equations with SIMPLE Method

## Introduction

This simulation solves the 2D steady incompressible Navier-Stokes equations using the SIMPLE (Semi-Implicit Method for Pressure Linked Equations) algorithm on a staggered finite volume grid:

**Continuity equation:**
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

**Momentum equations:**
$$\rho \left(u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}\right) = -\frac{\partial p}{\partial x} + \mu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

$$\rho \left(u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}\right) = -\frac{\partial p}{\partial y} + \mu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$

Where:
- $u, v$ = velocity components in x, y directions
- $p$ = pressure
- $\rho$ = density (constant for incompressible flow)
- $\mu$ = dynamic viscosity

### Numerical Method

The SIMPLE algorithm uses:
1. **Staggered grid**: Pressure at cell centers, velocities at cell faces
2. **Under-relaxation**: Factors $\omega_u$, $\omega_v$, $\omega_p$ for stability
3. **Iterative convergence**: Based on mass conservation and velocity/pressure residuals
4. **Geometric flexibility**: Supports various channel geometries with obstacles

### Convergence Criteria

The solution is considered converged when all criteria are met:
1. **Mass conservation**: $\left|\sum \text{mass flux}\right| < \text{mass\_tolerance}$
2. **Velocity convergence**: $\text{RMSE}(u) < \text{u\_rmse\_tolerance}$, $\text{RMSE}(v) < \text{v\_rmse\_tolerance}$
3. **Pressure convergence**: $\text{RMSE}(p) < \text{p\_rmse\_tolerance}$

## Test Cases

The solver supports multiple boundary conditions and geometries:

1. **p1 - Channel Flow**: Standard rectangular channel with uniform inlet velocity
   - Geometry: Straight channel
   - Boundary: Uniform inlet, no-slip walls, pressure outlet
   - Reynolds: Low to moderate

2. **p2 - Back Stair Flow**: Channel with backward-facing step
   - Geometry: Channel with sudden expansion
   - Boundary: Uniform inlet, no-slip walls, pressure outlet
   - Reynolds: Low to moderate

3. **p3 - Expansion Channel**: Channel with gradual expansion
   - Geometry: Diverging channel
   - Boundary: Uniform inlet, no-slip walls, pressure outlet
   - Reynolds: Low to moderate

4. **p4 - Cube Driven Flow**: Channel with cubic obstacle
   - Geometry: Channel with cubic blockage
   - Boundary: Uniform inlet, no-slip walls, pressure outlet
   - Reynolds: Low to moderate

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **Mesh Resolution Convergence Search (iterative+0-shot)**
   - **mesh_x**: Number of grid cells in x-direction, determines spatial resolution along channel length
   - **mesh_y**: Number of grid cells in y-direction, determines spatial resolution across channel width
   - Both use iterative refinement with multiplication factor of 2

2. **Under-Relaxation Factor Optimization (0-shot)**
   - **omega_u**: Under-relaxation factor for u-velocity (0.1 ≤ ω ≤ 1.0)
   - **omega_v**: Under-relaxation factor for v-velocity (0.1 ≤ ω ≤ 1.0)
   - **omega_p**: Under-relaxation factor for pressure (0.1 ≤ ω ≤ 0.5)

3. **Convergence Threshold Optimization (0-shot)**
   - **diff_u_threshold**: Convergence threshold for u-velocity iterations (1e-07 to 1e-03)
   - **diff_v_threshold**: Convergence threshold for v-velocity iterations (1e-07 to 1e-03)
   - **res_iter_v_threshold**: Residual threshold for inner v-velocity iterations (1e-07 to 1e-03 or exp_decay)

### Dummy Strategy

1. **Mesh Resolution Convergence Search (iterative+0-shot)**
   - **mesh_x**: Start from 64, multiply by 2 each iteration until convergence
   - **mesh_y**: Start from 16, multiply by 2 each iteration until convergence
   - **Non-target parameters**: Fixed relaxation factors (ω_u=0.6, ω_v=0.6, ω_p=0.3) and tight thresholds
   - **Aspect ratios**: Test with 4 out of 5 different aspect ratios (0.1, 0.2, 0.25, 0.5, 1.0) (first four for mesh_x and last four for mesh_y) for geometric sensitivity

2. **Under-Relaxation Factor Optimization (0-shot)**
   - **omega_u**: Grid search over [0.1, 1.0] with 10 equally spaced values
   - **omega_v**: Grid search over [0.1, 1.0] with 10 equally spaced values
   - **omega_p**: Grid search over [0.1, 0.5] with 8 equally spaced values
   - **Non-target parameters**: 4 paired mesh combinations [(64,16), (128,32), (192,48), (256,64)]

3. **Convergence Threshold Optimization (0-shot)**
   - **diff_u_threshold**: Grid search over [1e-07, 1e-03] with 5 values (decreasing order)
   - **diff_v_threshold**: Grid search over [1e-07, 1e-03] with 5 values (decreasing order)
   - **res_iter_v_threshold**: Grid search over [1e-07, 1e-03] with 5 values or exp_decay
   - **Non-target parameters**: 4 paired mesh combinations [(64,16), (128,32), (192,48), (256,64)]

### Wall Scaling Strategy

For mesh tasks, wall dimensions scale proportionally with mesh resolution:
- **wall_height**: Scales with mesh_y (maintains height/mesh_y ratio)
- **wall_width**: Scales with mesh_x (maintains width/mesh_x ratio)
- **wall_start_height**: Scales with mesh_y (maintains position/mesh_y ratio)
- **wall_start_width**: Scales with mesh_x (maintains position/mesh_x ratio)

Base wall values: height=4, width=16, start_height=4, start_width=20

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range | Search Type |
|-----------|-------------|-------|-------------|
| mesh_x | Number of grid cells in x-direction | 64 ≤ mesh_x ≤ 256 | iterative+0-shot |
| mesh_y | Number of grid cells in y-direction | 16 ≤ mesh_y ≤ 64 | iterative+0-shot |
| omega_u | Under-relaxation factor for u-velocity | 0.1 ≤ omega_u ≤ 1.0 | 0-shot |
| omega_v | Under-relaxation factor for v-velocity | 0.1 ≤ omega_v ≤ 1.0 | 0-shot |
| omega_p | Under-relaxation factor for pressure | 0.1 ≤ omega_p ≤ 0.5 | 0-shot |
| diff_u_threshold | Convergence threshold for u-velocity | 1e-07 ≤ diff_u_threshold ≤ 1e-03 | 0-shot |
| diff_v_threshold | Convergence threshold for v-velocity | 1e-07 ≤ diff_v_threshold ≤ 1e-03 | 0-shot |
| res_iter_v_threshold | Residual threshold for inner iterations | 1e-07 ≤ res_iter_v_threshold ≤ 1e-03 or exp_decay | 0-shot |

### Other

| Parameter | Description | Default Values by Profile |
|-----------|-------------|---------------------------|
| length | Channel length | p1: 20.0, p2: 14.56, p3: 12.8, p4: 10.23 |
| breadth | Channel width | p1: 1.0, p2: 1.14, p3: 1.28, p4: 1.2 |
| mu | Dynamic viscosity | p1: 0.01, p2: 0.04181, p3: 0.04448, p4: 0.00753 |
| rho | Density | p1: 1.0, p2: 4.42, p3: 3.91, p4: 1.3 |
| max_iter | Maximum iterations | 25-50 (precision dependent) |
| verbose | Enable verbose output | False |

### Notes

- **Aspect ratios**: Different aspect ratios (0.1, 0.2, 0.25, 0.5, 1.0) test geometric sensitivity
- **Mesh combinations**: Paired values [(64,16), (128,32), (192,48), (256,64)] maintain consistent aspect ratios
- **Wall scaling**: Wall dimensions scale proportionally with mesh resolution to maintain physical geometry
- **Convergence order**: Thresholds decrease from loose to tight during search (easier to harder convergence)
- **Relaxation factors**: Lower values = more stable but slower convergence, higher values = faster but potentially unstable

## Checkout

### Summary

- **Benchmarks**: 4 profiles (p1: channel_flow, p2: back_stair_flow, p3: expansion_channel, p4: cube_driven_flow)
- **Target Parameters**: 8 (mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold)
- **Precision Levels**: 3 (low, medium, high with varying convergence criteria)

### Task Distribution

Current configuration generates:

- **mesh_x** (iterative+0-shot): 4 profiles × 4 aspect ratios = 16 tasks per precision
- **mesh_y** (iterative+0-shot): 4 profiles × 4 aspect ratios = 16 tasks per precision
- **omega_u** (0-shot): 4 profiles × 4 mesh combinations = 16 tasks per precision
- **omega_v** (0-shot): 4 profiles × 4 mesh combinations = 16 tasks per precision
- **omega_p** (0-shot): 4 profiles × 4 mesh combinations = 16 tasks per precision
- **diff_u_threshold** (0-shot): 4 profiles × 4 mesh combinations = 16 tasks per precision
- **diff_v_threshold** (0-shot): 4 profiles × 4 mesh combinations = 16 tasks per precision
- **res_iter_v_threshold** (0-shot): 4 profiles × 4 mesh combinations = 16 tasks per precision
- **Total per precision**: 128 tasks
- **Total tasks**: 384 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/ns_channel_2d.yaml`
Cache script: `checkouts/ns_channel_2d.py`

### Key Features

1. **Geometric Flexibility**: Supports multiple channel geometries with obstacles
2. **Aspect Ratio Testing**: Evaluates sensitivity to different mesh aspect ratios
3. **Proportional Scaling**: Wall dimensions scale with mesh resolution
4. **Convergence Optimization**: Multiple precision levels for accuracy vs. cost trade-off
5. **Parameter Sensitivity**: Tests critical parameters (mesh, relaxation factors) and secondary parameters (thresholds)
