# Heat 1D Equations with Explicit Finite Difference Method

## Introduction

This simulation solves the 1D heat conduction equation with mixed boundary conditions using an explicit finite difference scheme:

**Heat Equation:**
$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$$

Where:

- $T$ = temperature
- $\alpha = \frac{k}{\rho c_p}$ = thermal diffusivity
- $k$ = thermal conductivity
- $\rho$ = density  
- $c_p$ = specific heat capacity

**Boundary Conditions:**

- Left boundary (x=0): Convection to ambient temperature: $\frac{k}{\Delta x}(T_1 - T_0) = h(T_0 - T_{\infty})$
- Right boundary (x=L): Adiabatic: $\frac{\partial T}{\partial x} = 0$

### Numerical Discretization

The spatial discretization uses explicit finite differences:

$$T_i^{n+1} = T_i^n + \frac{\alpha \Delta t}{(\Delta x)^2} (T_{i-1}^n - 2T_i^n + T_{i+1}^n)$$

The time step is constrained by the CFL condition for diffusion:
$$\Delta t = \frac{\text{CFL} \cdot (\Delta x)^2}{2\alpha}$$

### Boundary Treatment

- **Left boundary (convection)**: $T_0 = \frac{\frac{\Delta x}{k} T_1 + h T_{\infty}}{\frac{\Delta x}{k} + h}$
- **Right boundary (adiabatic)**: $T_{N} = T_{N-1}$

## Test Cases

The solver supports multiple profiles with varying material properties and conditions:

1. **p1** - Reference case:
   - L = 0.15 m, k = 1.0 W/m-K, h = 1000 W/m²-K
   - ρ = 1500 kg/m³, cp = 1000 J/kg-K
   - T_inf = 12°C, T_init = 25°C

2. **p2-p25** - Variations with different material properties and conditions

**Note for LLM Developers**: When generating natural language descriptions of the test cases, read the actual profile configuration files (run_configs/heat_1d/p*.yaml) to extract specific parameter values for each profile and create accurate, detailed descriptions.

The simulated results are considered correct if the relative RMSE meets the precision-dependent tolerance (high: 0.0001, medium: 0.001, low: 0.01) compared to reference solution.

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **CFL Convergence Search (iterative+0-shot)**
   - CFL number controls temporal stability for diffusion: $\Delta t = \frac{\text{CFL} \cdot (\Delta x)^2}{2\alpha}$

2. **n_space Convergence Search (iterative+0-shot)**  
   - n_space determines spatial resolution: $\Delta x = L / n\_space$

### Dummy Strategy

1. **CFL Convergence Search (iterative+0-shot)**
   - For dummy solution, halve CFL each round (multiplication factor: 0.5) starting from 1.0 until convergence
   - **Non-target parameters**: n_space ∈ {64, 256, 1024} (different starting resolutions)

2. **n_space Convergence Search (iterative+0-shot)**
   - For dummy solution, double n_space each iteration (multiplication factor: 2) starting from 64 until convergence
   - **Non-target parameters**: CFL = 0.25 (fixed stable value)

## Summarized parameter table for developer only (Not LLM)

| Parameter | Description | Range |
|-----------|-------------|-------|
| cfl | Courant-Friedrichs-Lewy number for temporal stability | 0 < cfl ≤ 1 |
| n_space | Number of spatial grid points | 64 ≤ n_space ≤ 2048 |

**Other Parameters:**

- L: Domain length [m]
- k: Thermal conductivity [W/m-K]
- h: Convection coefficient [W/m²-K]
- ρ: Density [kg/m³]  
- cp: Specific heat [J/kg-K]
- T_inf: Ambient temperature [°C]
- T_init: Initial temperature [°C]

## Checkout

### Summary

- **Profiles**: 25 (p1: reference, p2-p25: material/condition variations)
- **Target Parameters**: 2 (CFL, n_space)
- **Precision Levels**: 3 (high: 0.0001, medium: 0.001, low: 0.01)

### Task Distribution

- **CFL**: 25 profiles × 3 non-target combos = 75 tasks
- **n_space**: 25 profiles × 1 non-target combo = 25 tasks  
- **Total per precision**: 100 tasks
- **Total tasks**: 300 tasks (across 3 precision levels)

**Non-target parameter variations:**

- For CFL search: n_space ∈ {64, 256, 1024}
- For n_space search: cfl = 0.25

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/heat_1d.yaml`
Cache script: `checkouts/heat_1d.py`
