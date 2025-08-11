# Burgers 1D Equation with 2nd Order Roe Method

## Introduction

This simulation solves the 1D inviscid Burgers equation, which serves as a simplified model for compressible gas dynamics, using a 2nd order Roe method with generalized minmod limiter:

**Conservation form:**
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0$$

where $u$ is the conserved variable representing velocity in the Burgers equation context.

**Flux function:**
$$f(u) = \frac{u^2}{2}$$

### Spatial Discretization

The spatial discretization uses MUSCL reconstruction with blending parameter $k$:

$$u^L_{j+\frac{1}{2}} = u_j + \frac{1-k}{4} \delta^+ u_{j-\frac{1}{2}} + \frac{1+k}{4} \delta^- u_{j+\frac{1}{2}}$$

$$u^R_{j+\frac{1}{2}} = u_{j+1} - \frac{1+k}{4} \delta^+ u_{j+\frac{1}{2}} - \frac{1-k}{4} \delta^- u_{j+\frac{3}{2}}$$

where $k$ is a blending coefficient between central ($k=1$) and upwind ($k=-1$) scheme.

### Slope Limiting

The slope limiter uses a generalized minmod limiter:

$$\delta^- u_{j+\frac{1}{2}} = \text{minmod}\left(w\frac{u_j - u_{j-1}}{\Delta x}, \frac{u_{j+1} - u_j}{\Delta x}\right)$$

$$\delta^+ u_{j+\frac{1}{2}} = \text{minmod}\left(\frac{u_{j+1} - u_j}{\Delta x}, w\frac{u_{j+2} - u_{j+1}}{\Delta x}\right)$$

where $w$ is a generalized parameter that controls slope limiting strength ($w \geq 1$).

### Flux Computation

The interface flux is computed using the Roe approximate Riemann solver:

$$F_{j+\frac{1}{2}} = \frac{1}{2}[f(u^L) + f(u^R)] - \frac{1}{2}|a|(u^R - u^L)$$

where $a = \frac{1}{2}(u^L + u^R)$ is the Roe-averaged wave speed.

## Test Cases

The case key in the config file sets different initial conditions:

1. **sin** - Sinusoidal wave: $u(x,0) = \sin(2\pi x/L) + 0.5$
2. **rarefaction** - Rarefaction wave: $u(x,0) = -0.1$ for $x < L/2$, $u(x,0) = 0.5$ for $x \geq L/2$
3. **sod** - Modified Sod shock tube: $u(x,0) = 1.0$ for $x < L/2$, $u(x,0) = 0.1$ for $x \geq L/2$
4. **double_shock** - Two interacting shocks: $u(x,0) = 1.0$ for $x < L/3$, $u(x,0) = 0.5$ for $L/3 \leq x < 2L/3$, $u(x,0) = 0.1$ for $x \geq 2L/3$
5. **blast** - Interacting blast waves: $u(x,0) = \exp\left(-\frac{(x-L/4)^2}{2\sigma^2}\right) + 0.8\cdot\exp\left(-\frac{(x-3L/4)^2}{2\sigma^2}\right)$, where $\sigma = L/20$

The simulated results are considered correct if the relative RMSE and L∞ norms meet the precision-dependent tolerance compared to reference solution, and the solution satisfies the convergence criteria:

1. **Mass conservation**: the total integral remains constant over time
2. **Energy non-increasing**: the total energy $\int u^2 dx$ should not increase 
3. **Total Variation (TV) non-increasing**: enforces entropy stability
4. **Maximum principle satisfaction**: solution bounded by initial condition extrema

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **CFL Convergence Search (iterative+0-shot)**
   - CFL (Courant-Friedrichs-Lewy) number is defined as: $CFL = u_{max} \cdot \frac{\Delta t}{\Delta x}$ where $u_{max}$ is the maximum wave speed

2. **n_space Convergence Search (iterative+0-shot)**
   - n_space determines spatial resolution: $\Delta x = L / n\_space$, where $L$ is domain length

3. **k-Parameter Optimization (0-shot)**
   - Grid search over k ∈ [-1.0, 1.0] with 21 equally spaced values to find the optimal blending parameter

4. **w-Parameter Optimization (0-shot)**
   - Grid search over w ∈ [1.0, 2.0] with 11 equally spaced values to find the optimal limiter parameter

### Dummy Strategy

1. **CFL Convergence Search (iterative+0-shot)**
   - For dummy solution, this means halve CFL each round (multiplication factor: 0.5) starting from 1.0 until convergence
   - **Non-target parameters**: combinations of k∈{-1.0, 0.0, 1.0} and w∈{1.0, 1.5, 2.0}

2. **n_space Convergence Search (iterative+0-shot)**
   - For dummy solution, this means doubling n_space each iteration (multiplication factor: 2) starting from 256 until convergence
   - **Non-target parameters**: CFL=0.25, with combinations of k∈{-1.0, 0.0, 1.0} and w∈{1.0, 1.5, 2.0}

3. **k-Parameter Optimization (0-shot)**
   - For dummy solution, grid search the k that achieves convergence with minimum computational cost after CFL refinement
   - **Non-target parameters**: starting CFL=1.0, with w∈{1.0, 1.5, 2.0} combinations

4. **w-Parameter Optimization (0-shot)**
   - For dummy solution, grid search the w that achieves convergence with minimum computational cost after CFL refinement
   - **Non-target parameters**: starting CFL=1.0, with k∈{-1.0, 0.0, 1.0} combinations

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| cfl | Courant-Friedrichs-Lewy number for stability | 0 < cfl ≤ 1 |
| k | Blending parameter between central (k=1) and upwind (k=-1) fluxes | -1 ≤ k ≤ 1 |
| w | Parameter for generalized minmod limiter strength | w ≥ 1 |
| n_space | Number of grid cells for spatial discretization | 64 ≤ n_space ≤ 8192 |

More Notes:

- $w = 1$: standard minmod limiter (most dissipative)
- $w = 2$: superbee-like behavior (least dissipative)
- $w$ must not be smaller than 1 otherwise limiter symmetry will be broken
- $n\_space$ determines spatial resolution: $\Delta x = L / n\_space$ (smaller $\Delta x$ = finer grid = higher accuracy but higher cost)

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| L | Domain length | 2.0 |
| case | Initial condition type | "sin" |
| record_dt | Time interval between recordings | 0.1 |
| end_frame | Simulation end after certain number of frames | 10 |
| dump_dir | Directory for output files | "sim_res/burgers_1d/p1" |
| verbose | Enable verbose output | False |

## Checkout

### Summary

- **Benchmarks**:
  - **p1**: Sinusoidal wave (smooth solution evolution)
  - **p2**: Rarefaction wave (expansion wave dynamics)
  - **p3**: Modified Sod shock tube (shock formation)
  - **p4**: Double shock (shock-shock interaction)
  - **p5**: Blast waves (complex multi-shock interactions)
- **Target Parameters**: 4 (CFL, n_space, k, w)
- **Precision Levels**: 3 (low: RMSE≤0.01/L∞≤0.05, medium: RMSE≤0.005/L∞≤0.025, high: RMSE≤0.001/L∞≤0.01)

### Task Distribution

Current configuration generates:

- **CFL** (iterative+0-shot): 5 profiles × 9 non-target combos = 45 tasks
- **n_space** (iterative+0-shot): 5 profiles × 9 non-target combos = 45 tasks
- **k** (0-shot): 5 profiles × 3 non-target combos = 15 tasks
- **w** (0-shot): 5 profiles × 3 non-target combos = 15 tasks
- **Total per precision**: 120 tasks
- **Total tasks**: 360 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/burgers_1d_config.yaml`
Cache script: `checkouts/burgers_1d_dummy_generation.py`
