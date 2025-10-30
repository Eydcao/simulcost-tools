# 2D Finite Element Method (FEM) with Implicit Newton Solver

## Introduction

This simulation solves solid mechanics problems using the **implicit** Finite Element Method (FEM) with a Newton solver and line search for nonlinear elasticity. The FEM discretizes the continuous solid mechanics problem into a finite-dimensional system of equations solved at each time step.

**Governing Equations:**

The FEM solves the momentum conservation equation:

$$\rho \frac{\partial^2 \mathbf{u}}{\partial t^2} = \nabla \cdot \boldsymbol{\sigma} + \rho \mathbf{g}$$

Where:

- $\mathbf{u}$ = displacement field
- $\rho$ = density
- $\boldsymbol{\sigma}$ = stress tensor (Cauchy stress)
- $\mathbf{g}$ = gravitational acceleration

**Constitutive Relations:**

The stress tensor is computed using **Corotational Elasticity**:

$$\mathbf{P} = 2\mu (\mathbf{F} - \mathbf{R}) + \lambda (J - 1) \mathbf{F}^{-T}$$

Where:

- $\mathbf{P}$ = first Piola-Kirchhoff stress tensor
- $\mathbf{F}$ = deformation gradient
- $\mathbf{R}$ = rotation matrix from polar decomposition ($\mathbf{F} = \mathbf{R}\mathbf{S}$)
- $\mu = \frac{E}{2(1+\nu)}$ = shear modulus
- $\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}$ = Lame's first parameter
- $J = \det(\mathbf{F})$ = volume ratio
- $E$ = Young's modulus
- $\nu$ = Poisson's ratio

### Spatial Discretization

The FEM uses a triangle mesh to discretize the domain:

1. **Element Formulation**: Each triangle element maps from reference configuration to current configuration using shape functions
2. **Energy Functional**: Total potential energy is assembled from elastic strain energy and external work
3. **Force Computation**: Element forces are computed from stress tensor via virtual work principle
4. **Stiffness Matrix**: Element stiffness matrices are assembled into global system

### Temporal Discretization

The time integration uses an **implicit backward Euler scheme** with Newton-Raphson solver:

$$\mathbf{x}^{n+1} = \mathbf{x}^n + \Delta t \mathbf{v}^n + \Delta t^2 \mathbf{M}^{-1} \mathbf{f}(\mathbf{x}^{n+1})$$

Where:

- $\mathbf{x}^{n+1}$ = position at next time step (unknown)
- $\mathbf{v}^n$ = velocity at current time step
- $\mathbf{M}$ = mass matrix
- $\mathbf{f}$ = internal + external forces
- **Newton Convergence Criterion**: $\frac{|\Delta \mathbf{x}|}{\Delta t} < \text{newton\_v\_res\_tol}$, where $\Delta \mathbf{x}$ is the position correction from Newton iteration

The solver uses line search to ensure descent direction and improve convergence robustness.

## Test Cases

The solver supports three different simulation cases (profiles). All units are SI (meters, kg, seconds, Pa, etc.):

1. **p1 - Cantilever Beam Simulation:**
   - **Domain**: 10.0 × 2.0 m
   - **Material**: E = 1.0×10⁵ Pa, ν = 0.29, ρ = 2.0 kg/m³
   - **Gravity**: -9.8 m/s²
   - **Initial conditions**: Beam fixed at left edge (x < small tolerance), initially at rest (v=0)
   - **Time step**: dt = 0.0005 s (default)
   - **Newton tolerance**: newton_v_res_tol = 0.01 m/s
   - **End time**: 4.0 s (200 frames with record_dt = 0.02 s)
   - **Tests**: Cantilever beam bending under gravity with large deformation

2. **p2 - Vibration Bar Simulation:**
   - **Domain**: 25.0 × 1.0 m (effective material region starts at x=5.0 m)
   - **Material**: E = 100.0 Pa, ν = 0.0, ρ = 1.0 kg/m³
   - **Gravity**: 0.0 (no gravity - energy conserving)
   - **Initial conditions**: Sinusoidal velocity field $v_x = 0.75 \sin(0.5\pi x/L_x)$ where $L_x$ is effective length, left edge fixed (x < x_start + small tolerance)
   - **Time step**: dt = 0.005 s (default)
   - **Newton tolerance**: newton_v_res_tol = 0.01 m/s
   - **End time**: 40.0 s (200 frames with record_dt = 0.2 s)
   - **Tests**: 1D elastic wave propagation and compression dynamics

3. **p3 - Twisting Column Simulation:**
   - **Domain**: 2.0 × 10.0 m (tall column)
   - **Material**: E = 1.0×10⁵ Pa, ν = 0.3, ρ = 1.0 kg/m³
   - **Gravity**: 0.0 (no gravity - energy conserving)
   - **Initial conditions**: Rotational velocity field around center, amplitude increases with height $y$, bottom edge fixed (y < small tolerance)
   - **Initial velocity**: $v_x = -A \cdot (y - y_c) \cdot (y/L_y)$, $v_y = A \cdot (x - x_c) \cdot (y/L_y)$ where $A$ = 1.0 m/s
   - **Time step**: dt = 0.001 s (default)
   - **Newton tolerance**: newton_v_res_tol = 0.01 m/s
   - **End time**: 4.0 s (200 frames with record_dt = 0.02 s)
   - **Tests**: 2D rotational dynamics and energy conservation in twisting motion

## Convergence Metrics

The simulated results are evaluated using two types of metrics:

### Self-Checking Metrics (Individual Simulation Validation)

1. **Energy Conservation**:
   - Total energy variation over time: $\text{var} = \frac{\sigma(E_{\text{tot}})}{\text{mean}(|E_{\text{tot}}|) + \epsilon} < \text{var\_threshold}$
   - Where $E_{\text{tot}} = E_{\text{kinetic}} + E_{\text{elastic\_potential}}$
   - Threshold depends on precision level (high: 0.015, medium: 0.02, low: 0.05)
   - **Note**: Only **p2** (vibration_bar) and **p3** (twisting_column) conserve energy (gravity = 0). **p1** (cantilever) does not conserve energy due to gravity, so energy conservation check is skipped for p1.

2. **Positivity Preservation**: Kinetic and elastic potential energies must be ≥ 0

3. **Newton Convergence**: All Newton iterations must converge within max_newton_iter (default: 10)
   - Convergence criterion: $\frac{|\Delta \mathbf{x}|}{\Delta t} < \text{newton\_v\_res\_tol}$

### Comparison Metrics (Between Adjacent Parameter Sets)

When comparing two simulations with different parameter values:

1. **Energy L2 Relative Difference**:
   - For each energy type (kin, pot): $\text{diff}_i = \frac{||E_1^i - E_2^i||_2}{||E_1^i||_2 + ||E_2^i||_2 + \epsilon}$
   - Average: $\text{avg\_diff} = \text{mean}(\text{diff}_{\text{kin}}, \text{diff}_{\text{pot}})$

2. **Convergence Criterion**:
   - $\text{converged} = (\text{avg\_diff} < \text{energy\_tolerance}) \land \text{energy\_conserved}_1 \land \text{energy\_conserved}_2 \land \text{positivity}_1 \land \text{positivity}_2$
   - Where energy_tolerance depends on precision level (high: 0.003, medium: 0.01, low: 0.03)

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **dx Grid Resolution Search (iterative+0-shot)**
   - dx controls the mesh resolution: element size in the x-direction
   - Smaller dx improves accuracy but increases computational cost quadratically (more elements and DOFs)

2. **dt Time Step Search (iterative+0-shot)**
   - dt controls the time step size for temporal discretization
   - Smaller dt improves temporal accuracy and Newton convergence robustness but increases total number of steps

### Dummy Strategy

1. **dx Grid Resolution Search (iterative+0-shot)**
   - Halve dx each iteration (multiplication factor: 0.5) starting from profile-specific initial values until convergence
   - **Profile-specific initial values**: p1=0.5, p2=0.5, p3=0.5
   - **Multiplication factor**: 0.5, **Max iterations**: 4
   - **Non-target parameters**: dt∈{p1:[0.0005, 0.00025], p2:[0.005, 0.0025], p3:[0.001, 0.0005]}

2. **dt Time Step Search (iterative+0-shot)**
   - Halve dt each iteration (multiplication factor: 0.5) starting from profile-specific initial values until convergence
   - **Profile-specific initial values**: p1=0.0005, p2=0.005, p3=0.001
   - **Multiplication factor**: 0.5, **Max iterations**: 4
   - **Non-target parameters**: dx∈{p1:[0.5, 0.25], p2:[0.5, 0.25], p3:[0.5, 0.25]}

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| dx | Mesh resolution (element size in x-direction) | 0.05 ≤ dx ≤ 1.0 |
| dt | Time step size (seconds) | 0.0001 < dt < 0.01 |

More Notes:

- **dx**: Determines spatial resolution; smaller values improve accuracy but increase computational cost quadratically
- **dt**: Time step size for implicit integration; smaller values improve temporal accuracy and Newton convergence robustness

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| case | Simulation case type (assigned by profile) | "cantilever" (p1) |
| E | Young's modulus (Pa) | Profile-dependent |
| nu | Poisson's ratio | Profile-dependent |
| density | Material density (kg/m³) | Profile-dependent |
| max_newton_iter | Maximum Newton iterations per time step | 10 |
| newton_v_res_tol | Newton velocity residual tolerance (m/s) | 0.01 |
| mesh_scale | Mesh scaling factor | 1.0 |
| verbose | Enable verbose output | False |
| dump_dir | Directory for output files | "sim_res/fem2d/p1" |

## Checkout

### Summary

- **Benchmarks**:
  - **p1**: Cantilever beam - bending under gravity (non-energy-conserving)
  - **p2**: Vibration bar - 1D elastic wave propagation (energy-conserving)
  - **p3**: Twisting column - 2D rotational dynamics (energy-conserving)
- **Target Parameters**: 2 (dx, dt)
  - **iterative+0-shot**: dx, dt
- **Precision Levels**: 3 (high, medium, low)

### Task Distribution

Current configuration generates:

- **dx** (iterative+0-shot): 3 profiles × 2 non-target combos = 6 tasks
- **dt** (iterative+0-shot): 3 profiles × 2 non-target combos = 6 tasks
- **Total per precision**: 12 tasks
- **Total tasks**: 36 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/fem2d.yaml`
Cache script: `checkouts/fem2d.py`
