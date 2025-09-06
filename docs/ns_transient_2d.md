# Navier-Stokes Transient 2D Equations with Taichi-based Fluid Simulation

## Introduction

This simulation solves the 2D transient incompressible Navier-Stokes equations using a Taichi-based fluid simulation framework with configurable boundary conditions and numerical schemes:

**Continuity equation:**
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

**Momentum equations:**
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \frac{1}{Re} \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

$$\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \frac{1}{Re} \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$

Where:
- $u, v$ = velocity components in x, y directions
- $p$ = pressure
- $Re$ = Reynolds number
- $t$ = time

### Numerical Method

The simulation uses:
1. **Taichi framework**: High-performance GPU/CPU computation with automatic differentiation
2. **Staggered grid**: Pressure at cell centers, velocities at cell faces
3. **CIP (Constrained Interpolation Profile) scheme**: High-order advection scheme for stability
4. **Pressure correction**: SIMPLE-like algorithm with configurable relaxation factor
5. **Time stepping**: CFL-controlled time step for temporal stability
6. **Vorticity confinement**: Optional artificial viscosity for numerical stability

### Domain Configuration

- **Aspect Ratio**: Fixed at 2.0 (y/x), meaning the domain is twice as wide as it is tall
- **Domain Resolution**: x_resolution = 2 × resolution, y_resolution = resolution
- **CFL Calculation**: $\Delta t = \text{CFL} \times \Delta x$ where $\Delta x = 1/\text{resolution}$
- **Maximum Wall Time**: 1200 seconds (20 minutes) per simulation

### Convergence Criteria

The solution is considered converged when:

**Normalized velocity RMSE**: $\text{RMSE}(\|\vec{v}\|) < \text{norm\_rmse\_tolerance}$

## Test Cases

The solver supports 12 profiles with 6 different boundary conditions, each tested at two Reynolds numbers (Re=1000 and Re=100000):

### Boundary Conditions

**BC1 - Simple Circular Obstacle (p1, p2)**
- Single circular obstacle in center of channel
- Uniform inlet velocity, pressure outlet
- Clean flow separation and wake formation

**BC2 - Multiple Obstacles with Steps (p3, p4)**
- Complex maze-like geometry with multiple rectangular obstacles
- Stepped flow path with alternating obstacle placement
- Tests flow through complex geometric constraints

**BC3 - Random Circular Obstacles (p5, p6)**
- 100 randomly placed circular obstacles (seed=123 for reproducibility)
- Dense obstacle field testing flow through irregular patterns
- Tests robustness to geometric complexity

**BC4 - Dual Inlet/Outlet Configuration (p7, p8)**
- Two separate inlet streams (top and bottom)
- Single central outlet
- Tests flow mixing and interaction between streams

**BC5 - Complex Obstacle Array (p9, p10)**
- Dense array of rectangular obstacles in systematic pattern
- Multiple flow paths with varying widths
- Tests flow through highly constrained geometries

**BC6 - Dragon-Shaped Obstacle (p11, p12)**
- Complex artistic obstacle loaded from PNG image file
- Irregular, organic shape testing flow around complex boundaries
- Tests numerical robustness with highly irregular geometry

### Profile Organization

Each boundary condition is tested at two Reynolds numbers:
- **Low Reynolds (Re=1000)**: Laminar flow characteristics, smooth flow patterns
- **High Reynolds (Re=100000)**: Turbulent flow characteristics, complex vortical structures

**Profile Mapping:**
- p1, p2: BC1 (circular obstacle)
- p3, p4: BC2 (multiple obstacles with steps)  
- p5, p6: BC3 (random circular obstacles)
- p7, p8: BC4 (dual inlet/outlet)
- p9, p10: BC5 (complex obstacle array)
- p11, p12: BC6 (dragon-shaped obstacle)

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **Resolution Convergence Search (iterative+0-shot)**
   - **resolution**: Grid resolution determining spatial discretization quality
   - Uses iterative refinement with multiplication factor of 2
   - Tests spatial convergence for different geometries

2. **CFL Optimization (iterative+0-shot)**
   - **cfl**: Courant-Friedrichs-Lewy number controlling time step stability
   - Uses iterative search starting from 0.2, dividing by 2 each iteration
   - Minimum value of 0.05 for stability

3. **Relaxation Factor Optimization (0-shot)**
   - **relaxation_factor**: Pressure correction relaxation factor controlling convergence rate
   - Grid search over [0.8, 1.5] with 8 equally spaced values
   - Tests convergence rate vs. stability trade-off

4. **Residual Threshold Optimization (0-shot)**
   - **residual_threshold**: Pressure solver convergence threshold
   - Uses specific values [1e-1, 1e-2, 5e-3] for precision control
   - Tests accuracy vs. computational cost trade-off

### Dummy Strategy

1. **Resolution Convergence Search (iterative+0-shot)**
   - **resolution**: Start from 50, multiply by 2 each iteration until convergence
   - **Non-target parameters**: Fixed CFL=0.05, relaxation_factor=1.3, residual_threshold=1e-2
   - Tests spatial discretization convergence across all geometries

2. **CFL Optimization (iterative+0-shot)**
   - **cfl**: Use exact values [0.2, 0.1, 0.05] from YAML configuration
   - **Non-target parameters**: resolution ∈ [200, 400], relaxation_factor=1.3, residual_threshold=1e-2
   - Tests temporal stability across different resolutions

3. **Relaxation Factor Optimization (0-shot)**
   - **relaxation_factor**: Use exact values [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5] from YAML
   - **Non-target parameters**: resolution ∈ [200, 400], cfl=0.05, residual_threshold=1e-2
   - Tests convergence rate optimization

4. **Residual Threshold Optimization (0-shot)**
   - **residual_threshold**: Use exact values [1e-1, 1e-2, 5e-3] from YAML configuration
   - **Non-target parameters**: resolution ∈ [200, 400], cfl=0.05, relaxation_factor=1.3
   - Tests precision vs. cost optimization

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range | Search Type |
|-----------|-------------|-------|-------------|
| resolution | Grid resolution for spatial discretization | 50 ≤ resolution ≤ 800 | iterative+0-shot |
| cfl | Courant-Friedrichs-Lewy number for time step | 0.05 ≤ cfl ≤ 0.2 | iterative+0-shot |
| relaxation_factor | Pressure correction relaxation factor | 0.8 ≤ relaxation_factor ≤ 1.5 | 0-shot |
| residual_threshold | Pressure solver convergence threshold | 1e-3 ≤ residual_threshold ≤ 1e-1 | 0-shot |

### Other

| Parameter | Description | Default Values by Profile |
|-----------|-------------|---------------------------|
| boundary_condition | Boundary condition type | 1-6 (varies by profile) |
| reynolds_num | Reynolds number | 1000.0 or 100000.0 |
| advection_scheme | Advection scheme | "cip" |
| vorticity_confinement | Vorticity confinement coefficient | 0.0 |
| total_runtime | Total simulation time | 1.0 |
| no_dye | Disable dye visualization | False |
| cpu | Use CPU instead of GPU | False |
| visualization | Visualization level | 0 |

### Domain Configuration

| Parameter | Description | Value |
|-----------|-------------|-------|
| aspect_ratio | Domain aspect ratio (y/x) | 2.0 (fixed) |
| x_resolution | Horizontal grid resolution | 2 × resolution |
| y_resolution | Vertical grid resolution | resolution |
| dx | Grid spacing | 1 / resolution |
| dt | Time step | CFL × dx |
| max_wall_time | Maximum simulation time | 1200 seconds |

### Notes

- **Boundary conditions**: 6 distinct geometries (BC1-BC6) with increasing complexity
- **Reynolds number pairs**: Each BC tested at Re=1000 (laminar) and Re=100000 (turbulent)
- **Domain aspect ratio**: Fixed at 2.0 (width/height) for all simulations
- **CFL calculation**: Simplified as dt = CFL × dx (assumes max velocity ≈ 1.0)
- **CFL values**: Exact values [0.2, 0.1, 0.05] to avoid floating-point precision issues
- **Relaxation factors**: Exact values [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5] for precision control
- **Residual thresholds**: Specific values [1e-1, 1e-2, 5e-3] for targeted precision testing
- **Transient behavior**: All simulations are time-dependent requiring careful time step control
- **Maximum runtime**: 20 minutes wall time limit per simulation

## Checkout

### Summary

- **Profiles**: 12 (p1-p12 with varying geometries and Reynolds numbers)
- **Target Parameters**: 4 (resolution, cfl, relaxation_factor, residual_threshold)
- **Precision Levels**: 3 (high: 0.05, medium: 0.1, low: 0.5)

### Task Distribution

Current configuration generates:

- **resolution** (iterative+0-shot): 12 profiles × 1 non-target combo = 12 tasks per precision
- **cfl** (iterative+0-shot): 12 profiles × 2 non-target combos = 24 tasks per precision
- **relaxation_factor** (0-shot): 12 profiles × 2 non-target combos = 24 tasks per precision
- **residual_threshold** (0-shot): 12 profiles × 2 non-target combos = 24 tasks per precision
- **Total per precision**: 84 tasks
- **Total tasks**: 252 tasks (across 3 precision levels)

**Non-target parameter variations:**

- For resolution search: cfl=0.05, relaxation_factor=1.3, residual_threshold=1e-2
- For cfl search: resolution ∈ [200, 400], relaxation_factor=1.3, residual_threshold=1e-2
- For relaxation_factor search: resolution ∈ [200, 400], cfl=0.05, residual_threshold=1e-2
- For residual_threshold search: resolution ∈ [200, 400], cfl=0.05, relaxation_factor=1.3

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/ns_transient_2d.yaml`
Cache script: `checkouts/ns_transient_2d.py`

### Key Features

1. **Transient Simulation**: Time-dependent flow evolution requiring careful time step control
2. **Geometric Diversity**: 12 different geometries from simple to complex artistic shapes
3. **Reynolds Number Range**: Both laminar (Re=1000) and turbulent (Re=100000) regimes
4. **High-Performance Computing**: Taichi framework for GPU/CPU acceleration
5. **Precision Control**: Exact parameter values to avoid floating-point precision issues
6. **Convergence Optimization**: Multiple precision levels for accuracy vs. cost trade-off
7. **Parameter Sensitivity**: Tests critical parameters (resolution, CFL) and secondary parameters (relaxation, thresholds)
