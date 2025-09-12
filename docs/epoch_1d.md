# EPOCH 1D Simulation with Ionisation

## Introduction

This simulation utilizes a Particle in Cell (PIC) code called EPOCH to solve the 1D ionisation of a Carbon target irradiated with an ultra-intense laser pulse.

Initially the domain is loaded with pseudo-particles (that represent a number of real particles). The number of particles that are initially loaded into each cell is determined by the parameter npart

The general procedure that EPOCH and PIC methods follows

1. Based on the particles locations and velocities (allowing for the determination of $\rho$ and $\vec{j}$, where the current is calcualted via $\frac{\partial \rho}{\partial t} + \nabla \cdot \vec{j} = 0$ for charge conservation) interpolate onto the discrete grid determined by the parameter nx (which determines the number of grid cells). How the contributions of each particles are loaded onto the grid are based on the parameter particle_order (changing the shape of how many neighboring cells' properties are influenced by a particle in a specific cell due to the use of pseudo-particles)

2. Based on the interpolated $\rho$ and $\vec{j}$, Maxwell's Equations can be solved to find the electromagnetic fields on the grid.

$$
\nabla \cdot \vec{E}=\frac{\rho}{\epsilon_0}
$$
$$
\nabla \cdot \vec{B}=0
$$
$$
\nabla \times \vec{E}=-\frac{\partial \vec{B}}{\partial t}
$$
$$
\nabla \times \vec{B}=\mu_0(\vec{j}+\epsilon_0\frac{\partial \vec{E}}{\partial t})
$$

Solving these equations are done with a finite difference method controlled by the parameter field_order (determining how many points to use for each grid cell's field calculation), along with a timestep determeind by the parameter dt_multipler.

3. With the values of $\vec{E} (\vec{x},t)$ and $\vec{B} (\vec{x},t)$, the forces on the particles can be determined during the particle push as follows.

$$
\frac{d\vec{p}}{dt}=q(\vec{E} +\vec{v} \times \vec{B} )
$$,
$$
\frac{d\vec{x}}{dt}=\vec{v}=\frac{\vec{p}}{\gamma m}
$$
$$
\gamma=\sqrt{1+(\frac{|\vec{p}|}{mc})^2}
$$

Solving these equations are done with a finite difference method in time, where the finite time step is determined by the parameter dt_multipler (along with the grid size)

In this case, field ionisation is used to ionise the carbon target, where the carbon is ionised from the effects of the laser. When the simulation detects the necessary energy conditions for a pseduoparticle of Carbon to be ionised, it changes the charge of that Carbon particle as necessary and generates a new pseduoparticle for the ejected electron.

## Test Cases

We define 3 different combinations of laser amplitude and target density.

1. p1: Base Configuration: Define an $a_0=200$ and  $n_{target}=5n_{cr}$

2. p2: Weaker Laser: Define an $a_0=150$ and  $n_{target}=5n_{cr}$

3. p3: Denser Target: Define an $a_0=150$ and  $n_{target}=8n_{cr}$

Where:

* The normalized laser amplitude $a_0=|e|E_0/m_ec\omega_0$ ($|e|$ is the elementary charge of an electron, $E_0$ is the amplitude of the laser, $m_e$ is the mass of the electron, $c$ is the speed of light in a vacuum and $\omega_0=2\pi c / \lambda_0$ is the laser's frequency with wavelength $\lambda_0$)
* The critical plasma density $n_{cr}=\omega_0^2 m_e \epsilon_0 / |e|^2$

## Convergence criteria

The simulated results are considered correct if the [TODO Rohan, which kind of error of the which electric fields] meet the precision-dependent tolerance requirements [TODO Rohan, pls fullfill your 3 levels of accuracy requirements]

## Tuneable Parameters and Dummy Strategy

### Tasks

1. **nx**: Iterative+0-shot

* Defines the number of grid points used in the 1D simulation, where nx is an integer

2. **dt_multiplier**: 0-shot

* EPOCH uses a CFL-based time step calculation where dt_multiplier must be less than 1 for numerical stability.
$\Delta t = \frac{dt_{multiplier} \cdot \Delta x}{c}$
where $\Delta x$ is the spatial grid spacing and $c$ is the speed of light. Note that the above equation is valid only for 1D.

3. **npart**: Iterative+0-shot

* In the "species" block (listed as npart_per_cell), which defines the number of pseudoparticles to use in each cell. For this simulation, the only initial species is unionized carbon.

4. **field_order**: 0-shot

* In the control block, which defines the finite difference method used to solve Maxwell's Equations during each step. This parameter can either be 2, 4, or 6

5. **particle_order**: 0-shot

* Defines the particle weighting for the macro-particles. This parameter can either be 2, 3 or 5

### Dummy Strategy

1. **nx**: Iterative+0-shot

* Start with a value of 400 increase by 20% each iteration (with necessary rounding to convert to an integer) until convergence
* **Non-target Parameters**: dt_multiplier=0.95, npart=20, field_order ∈ {2,4,6}, particle_order ∈ {2,3,5}

2. **dt_multiplier**: 0-shot

* Grid search dt_multiplier between [0.80, 0.99] with 11 equally spaced values, varying nx for convergence check
* **Non-target Parameters**: npart=20, field_order ∈ {2,4,6}, particle_order ∈ {2,3,5}, nx=400 (starting)

3. **npart**: Iterative+0-shot

* Starting with 10 pseudoparticles per cell, increase number of particles by 20% (rounding as necessary to convert to an integer) until convergence
* **Non-target Parameters**: dt_multiplier=0.95, field_order ∈ {2,4,6}, particle_order ∈ {2,3,5}, nx=3200

4. **field_order**: 0-shot

* Vary the parameter between the possible field orders {2,4,6}, varying nx for convergence check
* **Non-target Parameters**: dt_multiplier=0.95, npart=20, particle_order ∈ {2,3,5}, nx=400 (starting)

5. **particle_order**: 0-shot

* Vary the parameter between the possible particle orders {2,3,5}, varying nx for convergence check
* **Non-target Parameters**: dt_multiplier=0.95, npart=20, field_order ∈ {2,4,6}, nx=400 (starting)

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| nx | Number of grid cells for spatial discretization | 400 ≤ nx ≤ 1600 |
| dt_multiplier | Scaling constant for temporal discretization | 0.80 ≤ dt_multiplier ≤ 0.99 |
| npart | Number of pseudoparticles per cell | 10 ≤ npart ≤ 35 |
| field_order | Field Integration Order | field_order ∈ {2,4,6} |
| particle_order | Particle Weighting Order | particle_order ∈ {2,3,5} |

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| $\lambda_0$ | Laser Wavelength | 1 $\mu$m  |
| t_end | End Time of Simulation | 800 fs |
| x_max | Domain Length | 160 $\mu$m |
| L_target | Thickness of Carbon Target | 30 $\mu$m |

## Output Data Structure

In general the simulation outputs .sdf (self-describing files) for various properties that can include:

* **Electric and Magnetic Field Data**: Spatial and temporal evolution of electromagnetic fields on the grid
* **Particle Data**: Positions, velocities, and energies of the pseudoparticles
* **Density Profiles**: Charge and current density distributions of all particles on the grid

**File Formats**: In `runners/epoch.py` 2 main parameters are outputted: $E_y$ and $n_e$ into HDF5 (.h5) files for field and particle data, with JSON metadata for run parameters and performance metrics.

## Checkout

### Summary

* Benchmarks (see test cases above for more details)

  * p1: Base Configuration
  * p2: Weaker Laser
  * p3: Denser target

* Target Parameters: 5 (nx, dt_multipler, npart, field_order, particle_order)
* Percision Levels: 3 (low at 0.36, medium at 0.33, high at 0.30)

### Task Distribution

* nx: 3 profiles x 9 non-target combos = 27 tasks
* dt_multipler: 3 profiles x 9 non-target combos = 27 tasks
* npart: 3 profiles x 9 non-target combos = 27 tasks
* field_order: 3 profiles x 3 non-target combos = 9 tasks
* particle_orderL 3 profiles x 3 non-target combos = 9 tasks

* Total per percision: 99 tasks
* Total tasks: 297 (across 3 percisions)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/epoch.yaml` Cache script: `checkouts/epoch.py`
