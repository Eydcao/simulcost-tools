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

$$\mathbf{U}^L_{j+\frac{1}{2}} = \mathbf{U}_j + \frac{1+k}{4} \psi(r_{j+\frac{1}{2}}) (\mathbf{U}_{j+2} - \mathbf{U}_{j+1}) + \frac{1-k}{4} \psi(r_{j-\frac{1}{2}}) (\mathbf{U}_{j+1} - \mathbf{U}_{j})$$

$$\mathbf{U}^R_{j+\frac{1}{2}} = \mathbf{U}_{j+1} - \frac{1+k}{4} \psi(r_{j+\frac{1}{2}}) (\mathbf{U}_{j+2} - \mathbf{U}_{j+1}) - \frac{1-k}{4} \psi(r_{j+\frac{3}{2}}) (\mathbf{U}_{j+3} - \mathbf{U}_{j+2})$$

where $k$ is a blending coefficient between central ($k=1$) and upwind ($k=-1$) scheme, and $\psi(r)$ is the slope limiter function.

## Slope Limiting

The slope limiter uses a generalized superbee limiter:

$$\psi(r) = \max\left[0, \max\left[\min(\beta r, 1), \min(r, \beta)\right]\right]$$

where $\beta$ is the limiter parameter controlling dissipation:
- $\beta = 1$: minmod limiter (most dissipative)
- $\beta = 2$: superbee limiter (least dissipative)

The slope ratio $r$ at interface $j+\frac{1}{2}$ is defined as:

$$r_{j+\frac{1}{2}} = \frac{\mathbf{U}_{j+1} - \mathbf{U}_{j}}{\mathbf{U}_{j+2} - \mathbf{U}_{j+1}}$$

This ratio compares the upwind slope to the local slope to detect smooth vs. discontinuous regions.

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

### Tasks

1. **CFL Convergence Search (0-shot and iterative)**
   - CFL (Courant-Friedrichs-Lewy) number is defined as: $CFL = \frac{(|u| + c) \Delta t}{\Delta x}$ where $c = \sqrt{\gamma p/\rho}$ is the speed of sound
   - For dummy solution, this means halve CFL each round until convergence (self checking criteria is $L^2 \leq 0.02$ and $L^{\infty} \leq 0.2$)
   - **Note**: This task spans different k selection (k=1,0,-1), plus β selections (β=0.5,1,2), representing different spatial schemes (ie, 9 CFL tasks)

```bash
# For example, for profile 1, the 9 tasks are:
# β=0.5, 3 different ks
python dummy_sols/euler_1d.py --profile p1 --task cfl --k 1 --beta 0.5
python dummy_sols/euler_1d.py --profile p1 --task cfl --k 0 --beta 0.5
python dummy_sols/euler_1d.py --profile p1 --task cfl --k -1 --beta 0.5
# β=1, 3 different ks
# ...
# β=2, 3 different ks
# ...
```

2. **β-Parameter (Composite: ie, 0-shot select β, then iteratively search CFL)**
   - Composite means the selection of β is 0-shot, but we still need to pay the cost for iteratively finding the convergent results
   - For dummy solution, this means:
     - For each β
     - Conduct a CFL search, and record the search cost
     - Record the β corresponding to the minimum search cost
   - **Note**: This task spans different k selection (k=-1,0,1), representing different spatial schemes (ie, 3 β tasks)

```bash
# For example, for profile 1, the 3 tasks are:
python dummy_sols/euler_1d.py --profile p1 --task beta --k -1
python dummy_sols/euler_1d.py --profile p1 --task beta --k 0
python dummy_sols/euler_1d.py --profile p1 --task beta --k 1
```

3. **k-Parameter (Composite: ie, 0-shot select k, then iteratively search CFL)**
   - Similar to the β-Parameter task, this is a composite task where k selection is 0-shot, followed by iterative CFL search
   - For dummy solution, this means:
     - For each k
     - Conduct a CFL search, and record the search cost
     - Record the k corresponding to the minimum search cost
   - **Note**: This task spans different β selection (β=0.5,1,2), representing different limiters (ie, 3 k tasks)

```bash
# For example, for profile 1, the 3 tasks are:
python dummy_sols/euler_1d.py --profile p1 --task k --beta 0.5
python dummy_sols/euler_1d.py --profile p1 --task k --beta 1
python dummy_sols/euler_1d.py --profile p1 --task k --beta 2
```

## Summarized parameter table for developer

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| cfl | Courant-Friedrichs-Lewy number for stability | 0 < cfl ≤ 1 |
| beta | Limiter parameter for generalized superbee | 0.5 ≤ beta ≤ 2 |
| k | Blending parameter between central (k=1) and upwind (k=-1) fluxes | -1 ≤ k ≤ 1 |

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| L | Domain length | 1.0 |
| gamma | Ratio of specific heats | 1.4 |
| case | Initial condition type | "sod" |
| n_space | Number of grid cells | 1024 |
| record_dt | Time interval between recordings | 0.02 |
| end_frame | Simulation end after certain number of frames | 10 |
| dump_dir | Directory for output files | "sim_res/euler_1d/p1" |
| verbose | Enable verbose output | False |