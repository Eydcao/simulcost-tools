# Unstructured Material Point Method (MPM) with Taichi-based Particle Simulation

## Introduction

This simulation solves solid mechanics problems using the Material Point Method (MPM) with Taichi-based particle simulation. The MPM is a hybrid Eulerian-Lagrangian method that combines the advantages of both approaches for simulating large deformation problems in solid mechanics.

**Governing Equations:**

The MPM solves the momentum conservation equation:

$$\frac{\partial \mathbf{v}}{\partial t} + \mathbf{v} \cdot \nabla \mathbf{v} = \frac{1}{\rho} \nabla \cdot \boldsymbol{\sigma} + \mathbf{g}$$

Where:
- $\mathbf{v}$ = velocity field
- $\rho$ = density
- $\boldsymbol{\sigma}$ = stress tensor
- $\mathbf{g}$ = gravitational acceleration

**Constitutive Relations:**

The stress tensor is computed using linear elasticity:

$$\boldsymbol{\sigma} = \frac{E}{1+\nu} \left( \boldsymbol{\epsilon} + \frac{\nu}{1-2\nu} \text{tr}(\boldsymbol{\epsilon}) \mathbf{I} \right)$$

Where:
- $E$ = Young's modulus
- $\nu$ = Poisson's ratio
- $\boldsymbol{\epsilon}$ = strain tensor
- $\mathbf{I}$ = identity tensor

### Spatial Discretization

The MPM uses a background Eulerian grid for solving the momentum equation and Lagrangian particles for tracking material properties:

1. **Particle-to-Grid Transfer**: Material properties are transferred from particles to grid nodes
2. **Grid Update**: Momentum equation is solved on the grid
3. **Grid-to-Particle Transfer**: Updated velocities are transferred back to particles
4. **Particle Update**: Particles are advected and their properties updated

### Temporal Discretization

The time integration uses an explicit scheme with CFL condition for stability:

$$\Delta t = \text{CFL} \cdot \frac{\Delta x}{\max(|\mathbf{v}| + c)}$$

Where $c$ is the wave speed in the material.

## Test Cases

The solver supports three different simulation cases (profiles):

1. **p1 - Cantilever Beam Simulation:**
   - Domain: 11.0 × 8.0 units
   - Material: E = 1.0×10⁵, ν = 0.29, ρ = 2.0
   - Gravity: -9.81
   - End time: 4.0 seconds
   - Tests beam bending under gravity

2. **p2 - Vibration Bar Simulation:**
   - Domain: 35.0 × 1.0 units (effective: 25.0 × 1.0)
   - Material: E = 100.0, ν = 0.0, ρ = 1.0
   - Gravity: 0.0 (no gravity)
   - End time: 40.0 seconds
   - Tests elastic wave propagation

3. **p3 - Disk Collision Simulation:**
   - Domain: 1.0 × 1.0 units
   - Material: E = 1000.0, ν = 0.3, ρ = 1000.0
   - Gravity: 0.0 (no gravity)
   - End time: 3.0 seconds
   - Tests impact dynamics

The simulated results are considered correct if they meet the precision-dependent energy tolerance and satisfy convergence criteria:

1. **Energy conservation**: Total energy should be conserved within tolerance
2. **Momentum conservation**: Linear and angular momentum should be conserved
3. **Physical realism**: Deformations should be physically reasonable

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **nx Grid Resolution Search (0-shot)**
   - nx controls the background grid resolution: $\Delta x = L / nx$ where $L$ is domain length
   - Higher resolution improves accuracy but increases computational cost
   - **Exact values**: [20, 40, 80, 100, 120]

2. **n_part Particle Density Search (iterative+0-shot)**
   - n_part controls the number of particles per grid cell
   - More particles improve material representation but increase computational cost
   - **Initial value**: 1, **Multiplication factor**: 2.0, **Max iterations**: 5

3. **CFL Stability Search (iterative+0-shot)**
   - CFL number controls time step size for temporal stability
   - **CRITICAL**: CFL should ALWAYS be less than 0.01 to avoid divergence
   - **Initial value**: 0.01, **Multiplication factor**: 0.5, **Max iterations**: 5
   - Smaller CFL improves stability but increases simulation time

4. **radii Optimization (0-shot)**
   - radii controls the support radius for particle interactions
   - **Range**: [1.3, 2.0] with 8 equally spaced values
   - Affects particle neighbor search and interaction strength

### Dummy Strategy

1. **nx Grid Resolution Search (0-shot)**
   - Test exact values [20, 40, 80, 100, 120] to find optimal resolution
   - **Non-target parameters**: n_part∈{2,4}, cfl=0.001, radii∈{1.5,2.0}

2. **n_part Particle Density Search (iterative+0-shot)**
   - Double n_part each iteration (multiplication factor: 2) starting from 1 until convergence
   - **Non-target parameters**: nx∈{50,100}, cfl=0.001, radii∈{1.5,2.0}

3. **CFL Stability Search (iterative+0-shot)**
   - Halve CFL each iteration (multiplication factor: 0.5) starting from 0.01 until convergence
   - **Non-target parameters**: nx∈{50,100}, n_part∈{2,4}, radii∈{1.5,2.0}
   - **WARNING**: CFL must be less than 0.01 to avoid divergence

4. **radii Optimization (0-shot)**
   - Grid search over radii∈[1.3, 2.0] with 8 equally spaced values to find optimal value
   - **Non-target parameters**: nx∈{50,100}, n_part∈{2,4}, cfl=0.001

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| nx | Background grid resolution (cells per unit length) | 20 ≤ nx ≤ 120 |
| n_part | Number of particles per grid cell | 1 ≤ n_part ≤ 32 |
| cfl | Courant-Friedrichs-Lewy number for temporal stability | 0 < cfl < 0.01 |
| radii | Support radius for particle interactions | 1.3 ≤ radii ≤ 2.0 |

More Notes:

- **nx**: Determines spatial resolution; higher values improve accuracy but increase computational cost quadratically
- **n_part**: Controls material representation; more particles improve accuracy but increase memory and computation
- **cfl**: **CRITICAL** - Must be less than 0.01 to avoid numerical divergence; smaller values improve stability
- **radii**: Controls particle interaction range; affects neighbor search and interaction strength

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
- **Target Parameters**: 4 (nx, n_part, cfl, radii)
- **Precision Levels**: 3 (high: 0.01, medium: 0.08, low: 0.3)

### Task Distribution

Current configuration generates:

- **nx** (0-shot): 3 profiles × 20 non-target combos = 60 tasks
- **n_part** (iterative+0-shot): 3 profiles × 20 non-target combos = 60 tasks
- **cfl** (iterative+0-shot): 3 profiles × 20 non-target combos = 60 tasks
- **radii** (0-shot): 3 profiles × 20 non-target combos = 60 tasks
- **Total per precision**: 240 tasks
- **Total tasks**: 720 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/unstruct_mpm.yaml`
Cache script: `checkouts/unstruct_mpm.py`

## Cost Analysis

The computational cost is tracked as:

- **Particle cost**: $n_{part}$ (number of particles per cell)
- **Communication cost**: $\sum_{each\_part} neighbor\_communication$ (inter-particle interactions)
- **Total cost**: $n_{part} + \sum_{each\_part} neighbor\_communication$

This provides a measure of computational work that scales with both particle density and inter-particle communication requirements. The cost calculation includes:

1. **Particle density cost**: Scales linearly with the number of particles per cell
2. **Neighbor communication cost**: Accounts for the computational overhead of particle-particle interactions within the support radius
3. **Spatial hash cost**: Includes the cost of spatial hashing for efficient neighbor search

The total cost provides a comprehensive measure of computational complexity that reflects both the discretization density and the interaction complexity in the unstructured MPM method.

## Important Notes for LLM Developers

**CRITICAL WARNING**: The CFL parameter must ALWAYS be less than 0.01 to avoid numerical divergence. This is a fundamental stability requirement for the MPM method and should be emphasized in all parameter optimization tasks.
