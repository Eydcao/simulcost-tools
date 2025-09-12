# EPOCH 1D Simulation with Ionisation

## Introduction

This simulation utilizes a Particle in Cell (PIC) code called EPOCH to solve the 1D ionisation of a Carbon target irradiated with an ultra-intense laser pulse.

**[PLACEHOLDER FOR ROHAN: Add physical system description, eg, governing equations, PIC algorithm details (eg, how many steps in PIC, which method we choose to use, how they are connected to some tunnable paramters (eg, interpolation orderings))]**

The general procedure that EPOCH uses is as follows:

1. Interpolate the electromagnetic fields from the grid onto the particles to find the forces experienced
2. Push the particles with the fields via a finite difference method
3. Calculate the electromagnetic fields on the grid based on the particles' new locations and velocities

## Test Cases

We define 3 different combinations of laser amplitude and target density.

1. p1: Base Configuration: Define an $a_0=200$ and  $n_{target}=5n_{cr}$

2. p2: Weaker Laser: Define an $a_0=150$ and  $n_{target}=5n_{cr}$

3. p3: Denser Target: Define an $a_0=150$ and  $n_{target}=8n_{cr}$

Where:

* The normalized laser amplitude $a_0=|e|E_0/m_ec\omega_0$ ($|e|$ is the elementary charge of an electron, $E_0$ is the amplitude of the laser, $m_e$ is the mass of the electron, $c$ is the speed of light in a vacuum and $\omega_0=2\pi c / \lambda_0$ is the laser's frequency with wavelength $\lambda_0$)
* The critical plasma density $n_{cr}=\omega_0^2 m_0 \epsilon_0 / |e|^2$

**[PLACEHOLDER FOR ROHAN: Add more detailed convergence criteria definition - what constitutes "correct" simulation results, tolerance levels, and physical validation requirements]**

The simulated results are considered correct if they meet the precision-dependent tolerance requirements and satisfy physical conservation laws and ionization physics.

## Tuneable Parameters and Dummy Strategy

Inside the input.deck file, one should see various "blocks" which contain important tuneable parameters.

1. **nx**: Iterative+0-shot

* In the "control" block, which defines the number of grid points used in the 1D simulation, where nx is an integer
* Start with a value of 400 increase by 20% each iteration (with necessary rounding to convert to an integer) until convergence
* **Non-target Parameters**: dt_multiplier=0.95, npart=20, field_order ∈ {2,4,6}, particle_order ∈ {2,3,5}

2. **dt_multiplier**: 0-shot

* In the "control" block, which adjusts the discrete time step. EPOCH uses a CFL-based time step calculation where dt_multiplier must be less than 1 for numerical stability.

$$\Delta t = \frac{dt_{multiplier} \cdot \Delta x}{c}$$

where $\Delta x$ is the spatial grid spacing and $c$ is the speed of light.

* Grid search dt_multiplier between [0.80, 0.99] with 11 equally spaced values, varying nx for convergence check
* **Non-target Parameters**: npart=20, field_order ∈ {2,4,6}, particle_order ∈ {2,3,5}, nx=400 (starting)

3. **npart**: Iterative+0-shot

* In the "species" block (listed as npart_per_cell), which defines the number of pseudoparticles to use in each cell. For this simulation, the only initial species is unionized carbon.
* Starting with 10 pseudoparticles per cell, increase number of particles by 20% (rounding as necessary to convert to an integer) until convergence
* **Non-target Parameters**: dt_multiplier=0.95, field_order ∈ {2,4,6}, particle_order ∈ {2,3,5}, nx=3200

4. **field_order**: 0-shot

* In the control block, which defines the finite difference method used to solve Maxwell's Equations during each step. This parameter can either be 2, 4, or 6
* Vary the parameter between the possible field orders {2,4,6}, varying nx for convergence check
* **Non-target Parameters**: dt_multiplier=0.95, npart=20, particle_order ∈ {2,3,5}, nx=400 (starting)

There is an additional parameter that can be adjusted when compiling the code

5. **particle_order**: 0-shot

* Defines the particle weighting for the macro-particles. This parameter can either be 2, 3 or 5
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

**[PLACEHOLDER FOR ROHAN: Add detailed description of EPOCH output files, data strucuture, key physical quantities' meaning, how to interprate the fig, etc]**

The simulation generates:

* **Electric and Magnetic Field Data**: Spatial and temporal evolution of electromagnetic fields
* **Particle Data**: Positions, velocities, and charge states of all particle species
* **Density Profiles**: Electron and ion density distributions
* **Energy Conservation**: Total electromagnetic and kinetic energy tracking
* **Ionization Diagnostics**: Ionization rates and charge state populations

**File Formats**: HDF5 (.h5) files for field and particle data, with JSON metadata for run parameters and performance metrics.

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
