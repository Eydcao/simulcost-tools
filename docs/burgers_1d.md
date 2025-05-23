# Burgers 1D Equation with 2nd Order Roe Method

This simulation solves the 1D inviscid Burgers equation using a 2nd order Roe method with minmod limiter:

$u_t + \left(\frac{u^2}{2}\right)_x = 0$

The Burgers equation is a fundamental PDE that exhibits nonlinear wave behavior including shock formation. It serves as a simplified model for compressible gas dynamics and is often used as a test case for numerical methods for conservation laws.

## Numerical Method

The solver uses a 2nd order Roe scheme with minmod limiter. Key components include:
- 2nd order spatial reconstruction using the minmod limiter
- Roe flux function for accurate representation of wave speeds
- Periodic boundary conditions
- Parameter k for blending between central (k=1) and upwind (k=-1) fluxes
- Parameter w for controlling the strength of the minmod limiter

## Initial Conditions

The solver supports several standard initial conditions for testing:

1. **sin** - Sinusoidal wave: u(x,0) = sin(2πx/L) + 0.5
3. **rarefaction** - Initial condition leading to symmetric rarefaction waves
4. **sod** - Modified Sod shock tube problem for Burgers equation
5. **double_shock** - Two interacting shock waves
6. **blast** - Interacting blast waves with Gaussian profiles

## Parameters and Tasks

### Tunable Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| cfl | Courant-Friedrichs-Lewy number for stability | 0 < cfl ≤ 1 |
| k | Blending parameter between central (k=1) and upwind (k=-1) fluxes | -1 ≤ k ≤ 1 |
| w | Parameter for minmod limiter strength | w ≥ 1 |

### Convergence Tasks

1. **CFL Convergence Search**
   - Start with CFL = 1.0
   - Halve the CFL value each iteration until solution convergence is achieved
   - A smaller CFL gives more accurate results but increases computational cost

2. **k-Parameter Optimization**
   - For each k value in range [-1, 1] (usually in steps of 0.2):
     - Find the largest convergent CFL number
     - Calculate computational cost for each (k, CFL) pair
   - Choose the k value that gives the lowest computational cost
   - Note: k=1 corresponds to a central scheme, k=-1 to a fully upwind scheme

3. **w-Parameter Selection**
   - For a given k value, any w between 1 and (3-k)/(1-k) is theoretically suitable
   - w=1 is typically a good default choice
   - Larger w values reduce the effect of the limiter (more accurate in smooth regions)

## Example Usage

```bash
# Run with default parameters
python runners/burgers_1d.py

# Specify custom parameters
python runners/burgers_1d.py cfl=0.5 k=0.5 w=1.2 n_space=512

# Choose a different initial condition
python runners/burgers_1d.py case=shock

# Use a different profile
python runners/burgers_1d.py --config-name=p2
```