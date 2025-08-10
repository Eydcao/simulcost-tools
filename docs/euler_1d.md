# Euler 1D Equations with 2nd Order MUSCL-Roe Method

This simulation solves the 1D Euler equations for compressible inviscid flow, using a 2nd order MUSCL scheme with Roe flux and generalized superbee limiter:

**Conservative form:**
$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = 0$$

Where the conservative variables and flux are:
$$\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho E \end{pmatrix}, \quad \mathbf{F} = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ u(\rho E + p) \end{pmatrix}$$

**Primitive variables:**
- $\rho$ = density
- $u$ = velocity  
- $p$ = pressure
- $E$ = specific total energy

**Equation of state:**
$$p = (\gamma - 1) \rho \left(E - \frac{u^2}{2}\right)$$

where $\gamma$ is the ratio of specific heats.

## Spatial Discretization

The spatial discretization uses MUSCL reconstruction with blending parameter $k$:

$$\mathbf{U}^L_{j+\frac{1}{2}} = \mathbf{U}_j + \frac{1+k}{4} \psi(r_{j}) (\mathbf{U}_{j+1} - \mathbf{U}_{j})$$

$$\mathbf{U}^R_{j+\frac{1}{2}} = \mathbf{U}_{j+1} - \frac{1+k}{4} \psi(r_{j+1}) (\mathbf{U}_{j+2} - \mathbf{U}_{j+1})$$

where $k$ is a blending coefficient between central ($k=1$) and upwind ($k=-1$) scheme, and $\psi(r)$ is the slope limiter function.

## Slope Limiting

The slope limiter uses a generalized superbee limiter:

$$\psi(r) = \max\left[0, \max\left[\min(\beta r, 1), \min(r, \beta)\right]\right]$$

where $\beta$ is the limiter parameter controlling dissipation.

The slope ratio $r$ at interface $j$ is defined as:

$$r_{j} = \frac{\mathbf{U}_{j+1} - \mathbf{U}_{j}}{\mathbf{U}_{j+2} - \mathbf{U}_{j+1}}$$

This ratio indicates the local non-smoothness, which will be the input into the slope limiter to achieve the TVD condition.

## Flux Computation

The interface flux is computed using the Roe approximate Riemann solver:

$$\mathbf{F}_{j+\frac{1}{2}} = \frac{1}{2}\left[\mathbf{F}(\mathbf{U}^L) + \mathbf{F}(\mathbf{U}^R)\right] - \frac{1}{2}|\mathbf{A}|(\mathbf{U}^R - \mathbf{U}^L)$$

where $|\mathbf{A}|$ is the Roe matrix with Roe-averaged quantities.

## Test Cases

The case key in the config file solver sets different initial conditions:

1. **sod** - Sod's shock tube problem: 
   - Left: $\rho=1.0, u=0.0, p=1.0$
   - Right: $\rho=0.125, u=0.0, p=0.1$

2. **TODO** - Additional test cases (lax, 123, etc.) to be added later

The simulated results are considered correct if both norms $L^2 \leq 0.02$ and $L^{\infty} \leq 0.2$ compared to reference solution, and the solution satisfies the convergence criteria:

1. **Mass conservation**: the total integral of density remains constant over time
2. **Energy conservation**: the total energy $\int \rho E \, dx$ should remain constant
3. **Positivity preservation**: pressure and density must remain positive at all times
4. **Shock speed consistency**: pressure gradients should not exceed physical bounds

## Parameter Tuning Tasks

**Principle**: When searching one parameter, all others must be fixed.

### Tasks

1. **CFL Convergence Search (0-shot and iterative)**
   - CFL (Courant-Friedrichs-Lewy) number is defined as: $CFL = \frac{(|u| + c) \Delta t}{\Delta x}$ where $c = \sqrt{\gamma p/\rho}$ is the speed of sound
   - For dummy solution, this means halve CFL each round until convergence (self checking criteria is $L^2 \leq 0.02$ and $L^{\infty} \leq 0.2$)
   - **All other parameters (β, k, n_space) must be specified and remain fixed**

```bash
# Example: CFL search with fixed β=1.0, k=1.0, n_space=256
python dummy_sols/euler_1d.py --profile p1 --task cfl --cfl 1.0 --beta 1.0 --k 1.0 --n_space 256
```

2. **n_space Convergence Search (0-shot and iterative)**
   - n_space determines spatial resolution: $\Delta x = L / n\_space$, where $L$ is domain length
   - This is a spatial convergence study: increase n_space (refine grid) until spatial convergence is achieved
   - For dummy solution, this means doubling n_space each iteration until convergence (self-checking criteria is $L^2 \leq 0.02$ and $L^{\infty} \leq 0.2$)
   - **All other parameters (CFL, β, k) must be specified and remain fixed**

```bash
# Example: n_space search with fixed CFL=0.25, β=1.0, k=1.0
python dummy_sols/euler_1d.py --profile p1 --task n_space --cfl 0.25 --beta 1.0 --k 1.0 --n_space 64
```

3. **β-Parameter Optimization (0-shot selection)**
   - Grid search over β ∈ [1.0, 2.0] to find the optimal limiter parameter
   - For each β value, iterate n_space until spatial convergence is achieved with **fixed CFL and k**
   - Select the β that achieves convergence with minimum computational cost
   - **CFL and k must be specified and remain fixed**

```bash
# Example: β optimization with fixed CFL=0.25, k=1.0
python dummy_sols/euler_1d.py --profile p1 --task beta --cfl 0.25 --k 1.0 --n_space 64
```

4. **k-Parameter Optimization (0-shot selection)**
   - Grid search over k ∈ [-1, 1] to find the optimal blending parameter
   - For each k value, iterate n_space until spatial convergence is achieved with **fixed CFL and β**
   - Select the k that achieves convergence with minimum computational cost
   - **CFL and β must be specified and remain fixed**

```bash
# Example: k optimization with fixed CFL=0.25, β=1.0  
python dummy_sols/euler_1d.py --profile p1 --task k --cfl 0.25 --beta 1.0 --n_space 64
```

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| cfl | Courant-Friedrichs-Lewy number for stability | 0 < cfl ≤ 1 |
| beta | Limiter parameter for generalized superbee | 1 ≤ beta ≤ 2 |
| k | Blending parameter between central (k=1) and upwind (k=-1) fluxes | -1 ≤ k ≤ 1 |
| n_space | Number of grid cells for spatial discretization | 64 ≤ n_space ≤ 2048 |

More Notes:
- $\beta = 1$: minmod limiter (most dissipative)
- $\beta = 2$: superbee limiter (least dissipative)
- $\beta$ must not be smaller than 1 otherwise symmetry will be broken
- When $k = -1$, $\beta$ no longer affects the solution
- $n\_space$ determines spatial resolution: $\Delta x = L / n\_space$ (smaller $\Delta x$ = finer grid = higher accuracy but higher cost)

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| L | Domain length | 1.0 |
| gamma | Ratio of specific heats | 1.4 |
| case | Initial condition type | "sod" |
| record_dt | Time interval between recordings | 0.02 |
| end_frame | Simulation end after certain number of frames | 10 |
| dump_dir | Directory for output files | "sim_res/euler_1d/p1" |
| verbose | Enable verbose output | False |
