# 1D Diffusion-Reaction Equations with Fully Implicit Newton Method

## Introduction

This simulation solves the 1D diffusion-reaction equation using a fully implicit Newton method with adaptive line search. The solver supports multiple reaction terms including Fisher-KPP, Allee effect, and Allen-Cahn (cubic) reactions.

**PDE form:**
$$\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} + f(u)$$

Where the reaction term $f(u)$ can be:

**Fisher-KPP reaction:**
$$f(u) = u(1-u)$$

**Allee effect reaction:**
$$f(u) = u(1-u)(u-a)$$

where $a$ is the Allee threshold parameter ($0 < a < 1$).

**Allen-Cahn (cubic) reaction:**
$$f(u) = u(1-u^2)$$

### Boundary and Initial Conditions

- **Boundary conditions:** Dirichlet conditions $u(0,t) = 1$ and $u(L,t) = 0$
- **Initial condition:** Step function $u(x,0) = 1$ for $0 \leq x \leq 2$, $u(x,0) = 0$ elsewhere
- **Domain:** $[0, L]$ where $L = 512$

### Spatial Discretization

The spatial discretization uses second-order central differences for the Laplacian:

$$\frac{\partial^2 u}{\partial x^2}\bigg|_i \approx \frac{u_{i-1} - 2u_i + u_{i+1}}{(\Delta x)^2}$$

where $\Delta x = L/n_{space}$ is the spatial step size.

### Temporal Discretization

The temporal discretization uses a fully implicit scheme:

$$\frac{u^{n+1} - u^n}{\Delta t} = \frac{\partial^2 u^{n+1}}{\partial x^2} + f(u^{n+1})$$

This results in a nonlinear system of equations that is solved using Newton's method.

### Newton Method Implementation

The Newton solver assembles the residual vector and Jacobian matrix:

**Residual (scaled by $\Delta x^2$ for stability):**
$$R_i = \frac{u_i^{n+1} - u_i^n}{\Delta t} - \frac{u_{i-1}^{n+1} - 2u_i^{n+1} + u_{i+1}^{n+1}}{(\Delta x)^2} - f(u_i^{n+1})$$

**Jacobian matrix elements:**
- Diagonal: $J_{i,i} = \frac{1}{\Delta t} + \frac{2}{(\Delta x)^2} - \frac{df}{du}(u_i^{n+1})$
- Off-diagonal: $J_{i,i\pm1} = -\frac{1}{(\Delta x)^2}$

### Line Search

The Newton method uses a greedy line search strategy:

1. Start with initial step size $\alpha = \alpha_0$
2. Try step: $u_{trial} = u + \alpha \cdot \Delta u$
3. If residual norm decreases, accept step; otherwise backtrack with $\alpha \leftarrow \alpha/2$
4. Continue until residual norm falls below tolerance or minimum step size is reached

## Test Cases

The solver supports three different reaction types (profiles):

1. **p1 - Fisher-KPP reaction:** $f(u) = u(1-u)$
   - Classic logistic growth with diffusion
   - Generates traveling wave solutions

2. **p2 - Allee effect reaction:** $f(u) = u(1-u)(u-a)$ with $a = 0.3$
   - Includes critical population threshold
   - Can lead to population extinction below threshold

3. **p3 - Allen-Cahn (cubic) reaction:** $f(u) = u(1-u^2)$
   - Phase field model with bistable potential
   - Generates interface dynamics

The simulated results are considered correct if they meet the precision-dependent tolerance (low: 0.15, medium: 5×10⁻⁴, high: 10⁻¹⁰) and satisfy convergence criteria:

1. **Newton convergence:** Residual norm falls below specified tolerance
2. **Physical bounds:** Solution remains bounded and physically reasonable
3. **Wave propagation:** For Fisher-KPP, traveling waves propagate at expected speeds

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **CFL Convergence Search (iterative+0-shot)**
   - CFL number controls time step size: $\Delta t = \text{CFL} \cdot (\Delta x)^2$ for diffusion stability
   - Higher CFL allows larger time steps but may reduce accuracy

2. **n_space Convergence Search (iterative+0-shot)**
   - n_space determines spatial resolution: $\Delta x = L / n_{space}$
   - Higher resolution improves accuracy but increases computational cost

3. **Tolerance Convergence Search (iterative+0-shot)**
   - Newton solver tolerance controls convergence criteria
   - Stricter tolerance improves solution accuracy but increases iterations

4. **Min Step Optimization (iterative+0-shot)**
   - Minimum step size for line search controls robustness
   - Smaller values improve convergence but may increase iterations

5. **Initial Step Guess Optimization (0-shot)**
   - Initial step size for line search controls aggressiveness
   - Larger values may converge faster but risk instability

### Dummy Strategy

1. **CFL Convergence Search (iterative+0-shot)**
   - Halve CFL each round (multiplication factor: 0.5) starting from 16.0 until convergence
   - **Non-target parameters:** n_space∈{512,1024,2048}, tol∈{10⁻⁶,10⁻⁷,10⁻⁸}, min_step=10⁻³, initial_step_guess=1.0

2. **n_space Convergence Search (iterative+0-shot)**
   - Double n_space each iteration (multiplication factor: 2) starting from 512 until convergence
   - **Non-target parameters:** cfl=0.5, tol∈{10⁻⁶,10⁻⁷,10⁻⁸}, min_step=10⁻³, initial_step_guess=1.0

3. **Tolerance Convergence Search (iterative+0-shot)**
   - Reduce tolerance by factor of 10 each iteration (multiplication factor: 0.1) starting from 10⁻⁵ until convergence
   - **Non-target parameters:** n_space∈{512,1024,2048}, cfl=0.5, min_step=10⁻³, initial_step_guess=1.0

4. **Min Step Optimization (iterative+0-shot)**
   - Reduce min_step by factor of 10 each iteration (multiplication factor: 0.1) starting from 10⁻¹ until convergence
   - **Non-target parameters:** n_space∈{512,1024,2048}, cfl=0.5, tol∈{10⁻⁶,10⁻⁷,10⁻⁸}, initial_step_guess=1.0

5. **Initial Step Guess Optimization (0-shot)**
   - Grid search over initial_step_guess∈{0.6,0.8,1.0,1.5,2.0,2.5} to find optimal value
   - **Non-target parameters:** n_space∈{512,1024,2048}, cfl=0.5, tol∈{10⁻⁶,10⁻⁷,10⁻⁸}, min_step=10⁻³

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| cfl | CFL number for temporal stability in diffusion equation | 0 < cfl ≤ 16 |
| n_space | Number of spatial grid points for discretization | 512 ≤ n_space ≤ 4096 |
| tol | Newton solver tolerance for convergence | 10⁻¹⁰ ≤ tol ≤ 10⁻⁵ |
| min_step | Minimum step size for Newton line search | 10⁻⁵ ≤ min_step ≤ 10⁻¹ |
| initial_step_guess | Initial step size for Newton line search | 0.6 ≤ initial_step_guess ≤ 2.5 |

More Notes:

- **CFL condition:** For diffusion equations, CFL ≤ 0.5 is typically stable, but higher values may work with implicit schemes
- **n_space:** Determines spatial resolution; higher values improve accuracy but increase computational cost quadratically
- **tol:** Stricter tolerance improves solution accuracy but increases Newton iterations; balance between accuracy and efficiency
- **min_step:** Controls line search robustness; too small may cause slow convergence, too large may cause instability
- **initial_step_guess:** Controls line search aggressiveness; larger values may converge faster but risk divergence

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| L | Domain length | 512.0 |
| reaction_type | Type of reaction term ("fisher", "allee", "cubic") | "fisher" |
| allee_threshold | Threshold parameter for Allee effect (only for allee type) | 0.3 |
| max_iter | Maximum Newton iterations per time step | 2000 |
| record_dt | Time interval between recordings | 25.6 |
| end_frame | Simulation end after certain number of frames | 10 |
| dump_dir | Directory for output files | "sim_res/diff_react_1d/p1" |
| verbose | Enable verbose output | False |

## Checkout

### Summary

- **Benchmarks**:
  - **p1**: Fisher-KPP reaction - traveling wave dynamics
  - **p2**: Allee effect reaction - critical threshold dynamics  
  - **p3**: Allen-Cahn (cubic) reaction - phase field interface dynamics
- **Target Parameters**: 5 (CFL, n_space, tol, min_step, initial_step_guess)
- **Precision Levels**: 3 (low: 0.15, medium: 5×10⁻⁴, high: 10⁻¹⁰)

### Task Distribution

Current configuration generates:

- **CFL** (iterative+0-shot): 3 profiles × 9 non-target combos = 27 tasks
- **n_space** (iterative+0-shot): 3 profiles × 9 non-target combos = 27 tasks
- **tol** (iterative+0-shot): 3 profiles × 9 non-target combos = 27 tasks
- **min_step** (iterative+0-shot): 3 profiles × 9 non-target combos = 27 tasks
- **initial_step_guess** (0-shot): 3 profiles × 9 non-target combos = 27 tasks
- **Total per precision**: 135 tasks
- **Total tasks**: 405 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/diff_react_1d.yaml`
Cache script: `checkouts/diff_react_1d.py`

## Cost Analysis

The computational cost is tracked as:

- **Newton cost:** $3 \times \text{total\_newton\_iters} \times n_{space}$
- **Line search cost:** $\text{total\_line\_search\_iters} \times n_{space}$
- **Total cost:** $\text{newton\_cost} + \text{line\_search\_cost}$

The factor of 3 in Newton cost accounts for:
1. Residual calculation
2. Jacobian assembly  
3. Linear system solve

This provides a measure of computational work that scales with both iteration count and problem size.
