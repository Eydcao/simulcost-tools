# CGYRO Simulation

## Introduction

This simulation utilizes the local-spectral gyrokinetic code CGYRO to solve the nonlinear gyrokinetic equations governing turbulent transport in magnetized fusion plasmas, enabling high-fidelity simulations of tokamak plasma turbulence in tokamaks. and stability.

The general procedure that CGYRO follows is:

1. The model is founded upon the Fokker-Plank–Maxwell system of equations, which are reduced via the gyrokinetic approximation under the assumption of strong magnetization and scale separation between equilibrium and fluctuations. The fast gyromotion is averaged out, yielding a five-dimensional phase-space system for the gyrocenter distribution function that is evolved in time. The resulting formulation retains parallel streaming, magnetic drift, turbulent drive, fluctuating electromagnetic fields, and collisional processes.

2. The resulting gyrokinetic system is discretized using a mixed spectral–grid approach. Perpendicular spatial directions are represented in Fourier space, enabling efficient evaluation of mode coupling. The coordinate parallel to the magnetic field is discretized on a field-aligned grid. Velocity space is resolved on a two-dimensional grid in the cosine of the pitch angle and magnetic the particle velocity. This procedure converts the continuous integro-differential system into a large set of coupled, time-dependent algebraic equations for each species and Fourier mode.

Although CGYRO uses a spectral representation in the spatial coordinates, a wave number advection algorithm allows the user to retain also global variations of the plasma radial pressure profiles.

3. The discretized system is advanced in time using an operator-splitting strategy. Collisionless terms are treated explicitly, while collisional terms implicitly. The nonlinear term is evaluated using a 2D Fast-Fourier transform (FFT) with dealiasing. At each timestep, moments of the distribution function are used to solve the electromagnetic Maxwell equations field equations enforcing quasineutrality and Ampère’s law, thereby updating the electromagnetic potentials. Collisional effects modelled with varying degrees of sophisticaton, from Lorentz to Sugama operators. The particle and field equations are thus evolved in a fully coupled, self-consistent manner.

4. Non-linear simulations are evolved until initial instabilities transition to a statistically stationary turbulent state, dubbed saturated. Particle, heat, and momentum fluxes, are computed as velocity and flux surface integrals. These quantities are time-averaged over the saturated phase to obtain transport coefficients and spectral characteristics of the turbulence.

The overall procedure consists of (i) reduction of the kinetic plasma description via gyrokinetic theory, (ii) mixed spectral–grid discretization, (iii) semi-implicit, operator-split time integration with self-consistent field solves, and (iv) statistical analysis of the resulting nonlinear turbulent state.

## Test Cases

We treat only linear simulations, thereby omitting non-linear mode coupling. 

We define 12 different profiles, each with their own combination of plasma configuration, scaled minor radius $rmin$, and normalized poloidal wavenumber $ky$. These cases here considered bear similarities with discharges realized on a large size tokamak, such as DIII-D. The most unstable linear mode in half of these cases propagates in the ion diamagnetic drift direction, while the other half propagates in the electron diamagnetic drift direction. Modes propagating in these two directions are commonly referred to ion and electron modes, respectively.

1. p1: Ion mode: $rmin=0.5$ and $ky=0.3$

2. p2: Electron mode: $rmin=0.7$ and $ky=0.3$

3. p3: Ion mode: $rmin=0.9$ and $ky=0.3$

4. p4: Ion mode: $rmin=0.9$ and $ky=0.05$

5. p5: Ion mode: $rmin=0.9$ and $ky=1.2$

6. p6: Electron mode: $rmin=0.9$ and $ky=2.5$

7. p7: Electron mode: $rmin=0.5$ and $ky=0.3$

8. p8: Ion mode: $rmin=0.7$ and $ky=0.3$

9. p9: Electron mode: $rmin=0.9$ and $ky=0.05$

10. p10: Electron mode: $rmin=0.9$ and $ky=0.3$

11. p11: Electron mode: $rmin=0.9$ and $ky=1.2$

12. p12: Ion mode: $rmin=0.9$ and $ky=2.5$

Where:

* The type of most unstable mode is determined by the sign of the real part $\omega$ of the converged eigenvalue.
* The scaled minor radius $rmin=r/a$ where $r$ is the minor radius and $a$ is the radius of the LCFS (Last Closed Flux Surface).
* The normalized poloidal wavenumber $ky$ defines the size of the instability under investigation in the direction binormal to the confining magnetic field line. CGYRO resolves instabilities at ion gyroradius scales, such as Trapped Electron (TEM) or Ion Temperature Gradient driven modes (ITG), as well as a electron gyroradius scales, viz. Electron Temperature gradient driven modes (ETG). The most unstable mode at any poloidal wave number is determined by the plasma configuration. 

## Convergence criteria

Simluations are terminated when the maximum simulation time is reached, or when the difference between eigenvalues across timesteps is below the tolerance threshold given by $freq_{tol}$. For a simulation with some proposed parameters, the results are considered correct when there is sufficiently small error between its returned eigenvalues and those of a reference solution.

## Tuneable Parameters and Dummy Strategy

### Tasks

1. **n_radial**: Iterative+0-shot

* Defines the number of radial points in the flux tube, which translates to the number of radial wavenumbers (radial Fourier harmonics) to retain in simulation, where $n_radial$ is an integer.

2. **n_theta**: Iterative+0-shot

* Defines the number of poloidal gridpoints, where $n_theta$ is an integer.

3. **n_xi**: Iterative+0-shot

* Defines the number of Legendre pseudospectral meshpoints to retain in simulation, where $n_xi$ is an integer.

4. **n_energy**: Iterative+0-shot

* Defines the number of generalized-Laguerre pseudospectral meshpoints to retain in simulation, where $n_energy$ is an integer.

5. **freq_tol**: Iterative+0-shot

* Defines the eigenvalue convergence tolerance for linear simulations, where $freq_tol$ is a positive real value.

6. **delta_t**: Iterative+0-shot

When using the adaptive method, the value of $delta_t$ is the size of the initial, or implicit, timestep. Then, the value of the explicit timestep is decreased adaptively to match the error tolerance, $error_{tol}$.

### Dummy Strategy

1. **n_radial**: Iterative+0-shot

* Start with a value of 4, increase by 200% each iteration (with necessary rounding to convert to an integer) until convergence
* **Non-target Parameters**: n_theta=24, n_xi=16, n_energy=8, error_tol=1e-4, freq_tol=1e-3, delta_t=1e-2

2. **n_theta**: Iterative+0-shot

* Start with a value of 6, increase by 200% each iteration (with necessary rounding to convert to an integer) until convergence
* **Non-target Parameters**: n_radial=8, n_xi=16, n_energy=8, error_tol=1e-4, freq_tol=1e-3, delta_t=1e-2

3. **n_xi**: Iterative+0-shot

* Start with a value of 6, increase by 200% each iteration (with necessary rounding to convert to an integer) until convergence
* **Non-target Parameters**: n_radial=8, n_theta=24, n_energy=8, error_tol=1e-4, freq_tol=1e-3, delta_t=1e-2

4. **n_energy**: Iterative+0-shot

* Start with a value of 4, increase by 200% each iteration (with necessary rounding to convert to an integer) until convergence
* **Non-target Parameters**: n_radial=8, n_theta=24, n_xi=16, error_tol=1e-4, freq_tol=1e-3, delta_t=1e-2

5. **freq_tol**: Iterative+0-shot

* Start with a value of 1e-5, increase by 1000% each iteration until convergence
* **Non-target Parameters**: n_radial=8, n_theta=24, n_xi=16, n_energy=8, error_tol=1e-4, delta_t=1e-2

6. **delta_t**: Iterative+0-shot

* Start with a value of 1e-2, increase by 200% each iteration until convergence
* **Non-target Parameters**: n_radial=8, n_theta=24, n_xi=16, n_energy=8, error_tol=1e-4, freq_tol=1e-3

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| n_radial | Number of radial wavenumberspoints (this sets the number of radial Fourier harmonics) | 4 ≤ n_radial ≤ 16 |
| n_theta | Number of poloidal gridpoints | 6 ≤ n_theta ≤ 24 |
| n_xi | Number of Legendre pseudospectral meshpoints | 6 ≤ n_xi ≤ 48 |
| n_energy | Number of generalized-Laguerre pseudospectral meshpoints | 4 ≤ n_energy ≤ 16 |
| freq_tol | Eigenvalue convergence tolerance for linear simulations | 1e-5 ≤ freq_tol ≤ 1e-3 |
| delta_t | Initial simulation timestep | 1e-2 ≤ delta_t ≤ 4e-2 |

### Other

Extensive documentation on the remaining input parameters used in CGYRO is available in the GACODE documentation at: https://gafusion.github.io/doc/cgyro/cgyro_list.html

## Output Data Structure

In general the simulation output data contains the following, among other information:

* **Eigenvalues**: Temporal evolution of the mode eigenvalue, this is used to determine convergence of linear simulations
Eigenvalues have a real and an imaginary part, which are called real frequency (omega) and growth rate (gamma). The sign of the real frequency often determines the nature of the mode, while the growth rate determines its strength.
* **Flux Data**: Flux of particles, momentum, and heat out of the plasma. These quantities are useful in non linear simulations


**File Formats**: In `runners/cgyro.py` the following main parameters are outputted: growth rates, mode frequencies, eigenvalues, and particle, heat, and momentum flux into HDF5 (.h5) files, with JSON metadata for run parameters and performance metrics.

## Checkout

### Summary

* Benchmarks (see test cases above for more details on the $rmin$ and $ky$ parameters)

  * p1: Ion mode: $rmin=0.5$ and $ky=0.3$
  * p2: Electron mode: $rmin=0.7$ and $ky=0.3$
  * p3: Ion mode: $rmin=0.9$ and $ky=0.3$
  * p4: Ion mode: $rmin=0.9$ and $ky=0.05$
  * p5: Ion mode: $rmin=0.9$ and $ky=1.2$
  * p6: Electron mode: $rmin=0.9$ and $ky=2.5$
  * p7: Electron mode: $rmin=0.5$ and $ky=0.3$
  * p8: Ion mode: $rmin=0.7$ and $ky=0.3$
  * p9: Electron mode: $rmin=0.9$ and $ky=0.05$
  * p10: Electron mode: $rmin=0.9$ and $ky=0.3$
  * p11: Electron mode: $rmin=0.9$ and $ky=1.2$
  * p12: Ion mode: $rmin=0.9$ and $ky=2.5$

* Target Parameters: 6 (n_radial, n_theta, n_xi, n_energy, freq_tol, delta_t)
* Precision Levels: 3 (low at 1e-3, medium at 1e-4, high at 1e-5)

### Task Distribution

* n_radial: 12 profiles = 12 tasks
* n_theta: 12 profiles = 12 tasks
* n_xi: 12 profiles = 12 tasks
* n_energy: 12 profiles = 12 tasks
* freq_tol: 12 profiles = 12 tasks
* delta_t: 12 profiles = 12 tasks

* Total per precision: 72 tasks
* Total tasks: 216 (across 3 precisions)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/cgyro.yaml` Cache script: `checkouts/cgyro.py`
