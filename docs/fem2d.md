# 2D Finite Element Method (FEM) with Implicit Newton Solver

## Introduction

This simulation solves solid mechanics problems using the **implicit** Finite Element Method (FEM) with a Newton solver and line search for nonlinear elasticity. The FEM discretizes the continuous solid mechanics problem into a finite-dimensional system of equations solved at each time step.

**Wall Time Constraint**: To prevent runaway simulations, a configurable wall time limit (default: 120 seconds) is enforced. Simulations that exceed this limit are terminated early and flagged as incomplete via the function call.

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
   - **End time**: 1.4 s
   - **energy_tolerance** (high: 0.005, medium: 0.010, low: 0.020)
   - **var_threshold** (high: 0.015, medium: 0.030, low: 0.060)
   - **Tests**: Cantilever beam bending under gravity with large deformation

2. **p2 - Vibration Bar Simulation:**
   - **Domain**: 25.0 × 1.0 m (effective material region starts at x=5.0 m)
   - **Material**: E = 100.0 Pa, ν = 0.0, ρ = 1.0 kg/m³
   - **Gravity**: 0.0 (no gravity - energy conserving)
   - **Initial conditions**: Sinusoidal velocity field $v_x = 0.75 \sin(0.5\pi x/L_x)$ where $L_x$ is effective length, left edge fixed (x < x_start + small tolerance)
   - **End time**: 5.6 s
   - **energy_tolerance** (high: 0.005, medium: 0.010, low: 0.020)
   - **var_threshold** (high: 0.030, medium: 0.060, low: 0.120)
   - **Tests**: 1D elastic wave propagation and compression dynamics

3. **p3 - Twisting Column Simulation:**
   - **Domain**: 2.0 × 10.0 m (tall column)
   - **Material**: E = 1.0×10⁵ Pa, ν = 0.3, ρ = 1.0 kg/m³
   - **Gravity**: 0.0 (no gravity - energy conserving)
   - **Initial conditions**: Rotational velocity field around center, amplitude increases with height $y$, bottom edge fixed (y < small tolerance)
   - **Initial velocity**: $v_x = -A \cdot (y - y_c) \cdot (y/L_y)$, $v_y = A \cdot (x - x_c) \cdot (y/L_y)$ where $A$ = 1.0 m/s
   - **End time**: 0.56 s
   - **energy_tolerance** (high: 0.010, medium: 0.020, low: 0.040)
   - **var_threshold** (high: 0.040, medium: 0.080, low: 0.160)
   - **Tests**: 2D rotational dynamics and energy conservation in twisting motion

4. **p4 - Gentle Vibration Bar Simulation:**
   - **Domain**: 25.0 × 1.0 m (effective material region starts at x=5.0 m)
   - **Material**: E = 100.0 Pa, ν = 0.0, ρ = 1.0 kg/m³
   - **Gravity**: 0.0 (no gravity - energy conserving)
   - **Initial conditions**: Sinusoidal velocity field $v_x = 0.1 \sin(0.5\pi x/L_x)$ where $L_x$ is effective length, left edge fixed (x < x_start + small tolerance)
   - **End time**: 5.6 s
   - **energy_tolerance** (high: 0.004, medium: 0.008, low: 0.016)
   - **var_threshold** (high: 0.012, medium: 0.024, low: 0.048)
   - **Tests**: 1D elastic wave propagation and compression dynamics with smaller amplitude

5. **p5 - Strong Twisting Column Simulation:**
   - **Domain**: 2.0 × 10.0 m (tall column)
   - **Material**: E = 1.0×10⁵ Pa, ν = 0.3, ρ = 1.0 kg/m³
   - **Gravity**: 0.0 (no gravity - energy conserving)
   - **Initial conditions**: Rotational velocity field around center, amplitude increases with height $y$, bottom edge fixed (y < small tolerance)
   - **Initial velocity**: $v_x = -A \cdot (y - y_c) \cdot (y/L_y)$, $v_y = A \cdot (x - x_c) \cdot (y/L_y)$ where $A$ = 2.5 m/s
   - **End time**: 0.56 s
   - **energy_tolerance** (high: 0.015, medium: 0.030, low: 0.060)
   - **var_threshold** (high: 0.060, medium: 0.120, low: 0.240)
   - **Tests**: 2D rotational dynamics and energy conservation in twisting motion with larger amplitude

## Convergence Metrics

The simulated results are evaluated using two types of metrics:

### Self-Checking Metrics (Individual Simulation Validation)

1. **Energy Conservation**:
   - Total energy variation over time: $\text{var} = \frac{\sigma(E_{\text{tot}})}{\text{mean}(|E_{\text{tot}}|) + \epsilon} < \text{var\_threshold}$
   - Where $E_{\text{tot}} = E_{\text{kinetic}} + E_{\text{elastic\_potential}}$
   - Threshold depends on precision level (high: 0.015, medium: 0.02, low: 0.05)

2. **Positivity Preservation**: Kinetic and elastic potential energies must be ≥ 0

### Comparison Metrics (Between Adjacent Parameter Sets)

When comparing two simulations with different parameter values:

1. **Energy L2 Relative Difference**:
   - For each energy type (kin, pot): $\text{diff}_i = \frac{||E_1^i - E_2^i||_2}{||E_1^i||_2 + ||E_2^i||_2 + \epsilon}$
   - Average: $\text{avg\_diff} = \text{mean}(\text{diff}_{\text{kin}}, \text{diff}_{\text{pot}})$

2. **Convergence Criterion**:
   - $\text{converged} = (\text{avg\_diff} < \text{energy\_tolerance}) \land \text{energy\_conserved}_1 \land \text{energy\_conserved}_2 \land \text{positivity}_1 \land \text{positivity}_2$

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **dx Grid Resolution Search (iterative+0-shot)**
   - dx controls the mesh resolution: element size in the x-direction
   - Smaller dx improves accuracy but increases computational cost quadratically (more elements and DOFs)

2. **cfl Search (iterative+0-shot)**
   - cfl controls the time step size for temporal discretization
   - Smaller cfl improves temporal accuracy and Newton convergence robustness but increases total number of steps

### Dummy Strategy

1. **dx Grid Resolution Search (iterative+0-shot)**
   - Halve dx each iteration (multiplication factor: 0.5) starting from profile-specific initial values until convergence
   - **Profile-specific initial values**: p1=0.5, p2=0.5, p3=0.5, p4=0.5, p5=0.5
   - **Multiplication factor**: 0.5, **Max iterations**: 4
   - **Non-target parameters**: cfl∈{[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]}

2. **cfl Search (iterative+0-shot)**
   - Halve cfl each iteration (multiplication factor: 0.5) starting from profile-specific initial values until convergence
   - **Profile-specific initial values**: p1=16.0, p2=16.0, p3=16.0, p4=16.0, p5=16.0
   - **Multiplication factor**: 0.5, **Max iterations**: 6
   - **Non-target parameters**: dx∈{[0.5, 0.25, 0.125, 0.0625]}

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| dx | Mesh resolution (element size in x-direction) | 0.05 ≤ dx ≤ 1.0 |
| cfl | CFL number for time step calculation | 0.1 ≤ cfl ≤ 16.0 |

More Notes:

- **dx**: Determines spatial resolution; smaller values improve accuracy but increase computational cost quadratically
- **cfl**: Controls the time step size; smaller values improve temporal accuracy and Newton convergence robustness

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
  - **p4**: Gentle vibration bar - 1D elastic wave propagation (energy-conserving)
  - **p5**: Strong twisting column - 2D rotational dynamics (energy-conserving)
- **Target Parameters**: 2 (dx, cfl)
  - **iterative+0-shot**: dx, cfl
- **Precision Levels**: 3 (high, medium, low)

### Task Distribution

Current configuration generates:

- **dx** (iterative+0-shot): 5 profiles × 6 non-target combos = 30 tasks
- **cfl** (iterative+0-shot): 5 profiles × 4 non-target combos = 20 tasks
- **Total per precision**: 50 tasks
- **Total tasks**: 150 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/fem2d.yaml`
Cache script: `checkouts/fem2d.py`
