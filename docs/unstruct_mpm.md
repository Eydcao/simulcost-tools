# Unstructured Material Point Method (MPM) with Taichi-based Particle Simulation

## Introduction

This simulation solves solid mechanics problems using the **explicit** Material Point Method (MPM) on unstructured mesh. The MPM is a hybrid Eulerian-Lagrangian method that combines the advantages of both approaches for simulating large deformation problems in solid mechanics.

**Governing Equations:**

The MPM solves the momentum conservation equation:

$$\frac{\partial \mathbf{v}}{\partial t} + \mathbf{v} \cdot \nabla \mathbf{v} = \frac{1}{\rho} \nabla \cdot \boldsymbol{\sigma} + \mathbf{g}$$

Where:

- $\mathbf{v}$ = velocity field
- $\rho$ = density
- $\boldsymbol{\sigma}$ = stress tensor
- $\mathbf{g}$ = gravitational acceleration

**Constitutive Relations:**

The stress tensor is computed using **Corotational Elasticity**:

$$\boldsymbol{\sigma} = 2\mu (\mathbf{F} - \mathbf{R}) \mathbf{F}^T + \lambda J(J-1)\mathbf{I}$$

Where:

- $\mathbf{F}$ = deformation gradient
- $\mathbf{R}$ = rotation matrix from polar decomposition ($\mathbf{F} = \mathbf{R}\mathbf{S}$)
- $\mu = \frac{E}{2(1+\nu)}$ = shear modulus
- $\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}$ = Lame's first parameter
- $J = \det(\mathbf{F})$ = volume ratio
- $E$ = Young's modulus
- $\nu$ = Poisson's ratio

### Spatial Discretization

The unstructured MPM uses a background triangle mesh for solving the momentum equation and Lagrangian particles for tracking material properties:

1. **Particle-to-Grid Transfer**: Material properties are transferred from particles to mesh vertices
2. **Grid Update**: Momentum equation is solved on the mesh
3. **Grid-to-Particle Transfer**: Updated velocities are transferred back to particles
4. **Particle Update**: Particles are advected and their properties updated

### Temporal Discretization

The time integration uses an explicit scheme with CFL condition for stability:

$$\Delta t = \frac{\text{CFL}}{v_{\text{max,init}} / \Delta x}$$

Where:

- $v_{\text{max,init}}$ = initial maximum velocity (upper bound, as the system has no new input forces)
- $\Delta x$ = characteristic grid spacing
- **Note**: this solver uses **explicit** material point methods and uses the initial maximum velocity to calcualted the dt, which is later as fixed for stability.

## Test Cases

The solver supports three different simulation cases (profiles). All units are SI (meters, kg, seconds, Pa, etc.):

1. **p1 - Cantilever Beam Simulation:**
   - **Domain**: 11.0 × 8.0 m
   - **Material**: E = 1.0×10⁵ Pa, ν = 0.29, ρ = 2.0 kg/m³
   - **Gravity**: -9.81 m/s²
   - **Initial conditions**: Beam fixed at left edge (x=0 to 1 m), positioned at y=5 to 7 m, initially at rest (v=0)
   - **Initial max velocity**: 0.5 m/s (used for CFL calculation)
   - **End time**: 4.0 s
   - **Tests**: Beam bending and large deformation under gravity

2. **p2 - Vibration Bar Simulation:**
   - **Domain**: 35.0 × 1.0 m (effective material region: 25.0 × 1.0 m starting at x=5 m)
   - **Material**: E = 100.0 Pa, ν = 0.0, ρ = 1.0 kg/m³
   - **Gravity**: 0.0 (no gravity)
   - **Initial conditions**: Sinusoidal velocity field $v_x = 0.75 \sin(0.5\pi x/L_x)$ where $L_x$ is effective length
   - **Initial max velocity**: 1.0 m/s (used for CFL calculation)
   - **End time**: 40.0 s
   - **Tests**: Elastic wave propagation and vibration modes

3. **p3 - Disk Collision Simulation:**
   - **Domain**: 1.0 × 1.0 m
   - **Material**: E = 1000.0 Pa, ν = 0.3, ρ = 1000.0 kg/m³
   - **Gravity**: 0.0 (no gravity)
   - **Initial conditions**: Two disks (radius R=0.2 m) moving toward each other with velocities (0.1, 0.1) and (-0.1, -0.1) m/s
   - **Initial max velocity**: 0.025 m/s (used for CFL calculation, accounting for collision dynamics)
   - **End time**: 3.0 s
   - **Tests**: Impact dynamics and contact resolution

## Convergence Metrics

The simulated results are evaluated using two types of metrics:

### Self-Checking Metrics (Individual Simulation Validation)

1. **Energy Conservation**:
   - Total energy variation over time using hybrid approach:
     - When mean energy is small (< 1e-10): $\text{var} = \sigma(E_{\text{tot}}) < \text{var\_threshold}$ (absolute threshold)
     - When mean energy is significant: $\text{var} = \frac{\sigma(E_{\text{tot}})}{\text{mean}(|E_{\text{tot}}|)} < \text{var\_threshold}$ (relative threshold)
   - Where $E_{\text{tot}} = E_{\text{kinetic}} + E_{\text{potential}} + E_{\text{gravitational}}$
   - Threshold depends on precision level (high: 0.015, medium: 0.02, low: 0.05)
   - Special handling for **disk_collision** case: only checks first 50% of time horizon as in this case the disks will collide each other causing diffusion.

2. **Positivity Preservation**: kinetic and elastic potential energies must be ≥ 0

3. **Wall Time Limit**: Simulation must complete within 600 seconds (10 minutes).  
   - This is checked internally by the solver: if the simulation exceeds the wall time limit during execution, it is immediately marked as **not converged** and stopped.

### Comparison Metrics (Between Adjacent Parameter Sets)

When comparing two simulations with different parameter values:

1. **Energy L2 Relative Difference**:
   - For each energy type (pot, kin, gra): $\text{diff}_i = \frac{||E_1^i - E_2^i||_2}{||E_1^i||_2 + ||E_2^i||_2 + \epsilon}$
   - Average: $\text{avg\_diff} = \text{mean}(\text{diff}_{\text{pot}}, \text{diff}_{\text{kin}}, \text{diff}_{\text{gra}})$
   - Note: Total energy (tot) is excluded from comparison to focus on individual energy components

2. **Convergence Criterion**:
   - $\text{converged} = (\text{avg\_diff} < \text{energy\_tolerance}) \land \text{energy\_conserved}_1 \land \text{energy\_conserved}_2$
   - Where energy_tolerance depends on precision level (high: 0.003, medium: 0.01, low: 0.03)

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **nx Grid Resolution Search (iterative+0-shot)**
   - nx controls the background grid resolution: $\Delta x = L / nx$ where $L$ is domain length
   - Higher resolution improves accuracy but increases computational cost

2. **n_part Particle Density Search (iterative+0-shot)**
   - n_part controls the number of particles per grid cell
   - More particles improve material representation but increase computational cost

3. **CFL Stability Search (iterative+0-shot)**
   - CFL number controls time step size for temporal stability
   - **CRITICAL**: CFL should ALWAYS be less than 0.01 to avoid divergence
   - Smaller CFL improves stability but increases simulation time

### Dummy Strategy

1. **nx Grid Resolution Search (iterative+0-shot)**
   - Double nx each iteration (multiplication factor: 2) starting from profile-specific initial values until convergence
   - **Profile-specific initial values**: p1=22, p2=35, p3=40
   - **Multiplication factor**: 2, **Max iterations**: 4
   - **Non-target parameters**: n_part∈{2,4}, cfl=0.001

2. **n_part Particle Density Search (iterative+0-shot)**
   - Double n_part each iteration (multiplication factor: 2) starting from 1 until convergence
   - **Initial value**: 1, **Multiplication factor**: 2.0, **Max iterations**: 5
   - **Non-target parameters**: nx∈{p1:[22,44], p2:[35,70], p3:[40,80]}, cfl=0.001

3. **CFL Stability Search (iterative+0-shot)**
   - Halve CFL each iteration (multiplication factor: 0.5) starting from 0.01 until convergence
   - **Initial value**: 0.01, **Multiplication factor**: 0.5, **Max iterations**: 5
   - **Non-target parameters**: nx∈{p1:[22,44], p2:[35,70], p3:[40,80]}, n_part∈{2,4}

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| nx | Background grid resolution (cells per unit length) | 20 ≤ nx ≤ 300 |
| n_part | Number of particles per grid cell | 1 ≤ n_part ≤ 32 |
| cfl | Courant-Friedrichs-Lewy number for temporal stability | 0.0001 < cfl < 0.01 |

More Notes:

- **nx**: Determines spatial resolution; higher values improve accuracy but increase computational cost quadratically
- **n_part**: Controls material representation; more particles improve accuracy but increase memory and computation
- **cfl**: **CRITICAL** - Should ALWAYS be less than 0.01 to avoid numerical divergence; smaller values improve stability
- **radii**: Fixed at 1.0 (support radius for particle interactions); not a tunable parameter

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| case | Simulation case type | "cantilever" |
| advect_scheme | Advection scheme type | 0 |
| flip_ratio | Blending parameter between PIC and FLIP methods | 1.0 |
| device | Computing device | "gpu" |
| verbose | Enable verbose output | 0 |
| dump_dir | Directory for output files | "sim_res/unstruct_mpm/p1" |

## Checkout

### Summary

- **Benchmarks**:
  - **p1**: Cantilever beam simulation - beam bending under gravity
  - **p2**: Vibration bar simulation - elastic wave propagation
  - **p3**: Disk collision simulation - impact dynamics
- **Target Parameters**: 3 (nx, n_part, cfl)
- **Precision Levels**: 3 (high, medium, low)

### Task Distribution

Current configuration generates:

- **nx** (iterative+0-shot): 3 profiles × 2 non-target combos = 6 tasks
- **n_part** (iterative+0-shot): 3 profiles × 2 non-target combos = 6 tasks
- **cfl** (iterative+0-shot): 3 profiles × 4 non-target combos = 12 tasks
- **Total per precision**: 24 tasks
- **Total tasks**: 72 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/unstruct_mpm.yaml`
Cache script: `checkouts/unstruct_mpm.py`

## Cost Analysis

The computational cost is tracked as a measure of computational complexity:

$$\text{Total Cost} = \sum_{\text{iteraton}}n_{\text{particles}} + \sum_{\text{each particle}} n_{\text{neighbor communications}}$$

Where:

- $n_{\text{particles}}$ = total number of particles in the simulation
- $n_{\text{neighbor communications}}$ = number of particle-particle interactions for each particle within the support radius

This cost metric captures:

1. **Particle Density Cost**: Scales with total number of particles
2. **Neighbor Communication Cost**: Accounts for particle-particle interactions within support radius (controlled by `radii` parameter, fixed at 1.0)
3. **Spatial Complexity**: Reflects both discretization density (via `nx` and `n_part`) and interaction complexity (via neighbor search on unstructured mesh)

The total cost provides a comprehensive measure of computational work that reflects both the resolution and the interaction complexity in the unstructured MPM method.
