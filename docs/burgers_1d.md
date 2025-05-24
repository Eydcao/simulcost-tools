# Burgers 1D Equation with 2nd Order Roe Method

This simulation solves the 1D inviscid Burgers equation, which serves as a simplified model for compressible gas dynamics, using a 2nd order Roe method with minmod limiter:

$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0$

Where, our spatial discretization reads the following

$u^l_{j+\frac{1}{2}} = u_j + \frac{1-\kappa}{4} \delta^+ u_{j-\frac{1}{2}} + \frac{1+\kappa}{4} \delta^- u_{j+\frac{1}{2}}$

$u^r_{j+\frac{1}{2}} = u_{j+1} - \frac{1+\kappa}{4} \delta^+ u_{j+\frac{1}{2}} - \frac{1-\kappa}{4} \delta^- u_{j+\frac{3}{2}}$

where $\kappa$ is a blending coefficient between central ($\kappa=1$) and upwind ($\kappa=-1$) scheme

The slope reads as below

$\delta^- u_{j+\frac{1}{2}} = \text{minmod}\left(\omega\frac{u_j - u_{j-1}}{\Delta x}, \frac{u_{j+1} - u_j}{\Delta x}\right)$

$\delta^+ u_{j+\frac{1}{2}} = \text{minmod}\left(\frac{u_{j+1} - u_j}{\Delta x}, \omega\frac{u_{j+2} - u_{j+1}}{\Delta x}\right)$

where $\omega$ is a generalized parameter that control the further side slope in minmod slope calculation

The case key in the config file solver sets different kinds of initial conditions for simulating:

1. **sin** - Sinusoidal wave: $u(x,0) = \sin(2\pi x/L) + 0.5$
2. **rarefaction** - Initial condition leading to unsymmetric rarefaction waves: $u(x,0) = -0.1 \text{ for } x < L/2, u(x,0) = 0.5 \text{ for } x \geq L/2$
3. **sod** - Modified Sod shock tube problem for Burgers equation: $u(x,0) = 1.0 \text{ for } x < L/2, u(x,0) = 0.1 \text{ for } x \geq L/2$
4. **double_shock** - Two interacting shock waves: $u(x,0) = 1.0 \text{ for } x < L/3, u(x,0) = 0.5 \text{ for } L/3 \leq x < 2L/3, u(x,0) = 0.1 \text{ for } x \geq 2L/3$
5. **blast** - Interacting blast waves with Gaussian profiles: $u(x,0) = \exp\left(-\frac{(x-L/4)^2}{2\sigma^2}\right) + 0.8\cdot\exp\left(-\frac{(x-3L/4)^2}{2\sigma^2}\right)$, where $\sigma = L/20$

The simulated results is considered to be correct if both the norms $L^2 \leq 12.5e-3$ and $L^{\infty} \leq 12.5e-2$ compared to reference solution, and the solution satisfy the below convergence criteria:

1. Mass conservation: the total integral of the solution remains constant over time
2. Energy non-increasing: the total energy $\int u^2 dx$ should not increase between consecutive time steps
3. Total Variation (TV) non-increasing: the total variation $\sum_i |u_{i+1} - u_i|$ should not increase over time, enforcing entropy stability
4. Maximum principle satisfaction: the solution values at all times must remain bounded by the initial condition's minimum and maximum values

## Parameter Tuning Tasks

### Tasks

1. **CFL Convergence Search (0-shot and iterative)**
   - CFL (Courant-Friedrichs-Lewy) number is defined as: $CFL = u_{max} \cdot \frac{dt}{dx}$ where $u_{max}$ is the maximum wave speed (which equals the maximum value of |u| for Burgers equation)
   - For dummy solution, this means halve cfl each round until convergence (self checking criteria is $L^2 \leq 5e-3$ and $L^{\infty} \leq 5e-2$), and the cost for iterative is the total cost for search, for 0-shot, is the cost of the corresponding cfl.
   - **Note**: This task spans different k selection (k=1,0,-1), pus w selections (w=1,1.5,2), representing different spatial schemes (ie, 9 CFL tasks)

```bash
# For example, for profile 1, the 9 tasks are:
# w=1, 3 different ks
python dummy_sols/burgers_1d.py --profile p1 --task cfl --k 1 --w 1
python dummy_sols/burgers_1d.py --profile p1 --task cfl --k 0 --w 1
python dummy_sols/burgers_1d.py --profile p1 --task cfl --k -1 --w 1
# w=1.5, 3 different ks
# ...
# w=2, 3 different ks
# ...
```

2. **k-Parameter (Composite: ie, 0-shot select k, then iteratively search CFL)**
   - Composite means the selection of k is 0-shot, but we still need to pay the cost for iteratively finding the convergent results; however, different k may influence the final cfl found, hence the total cost for cfl search.
   - However, we do not need to search for k, once we got a converged result (ie, task is zero-shot wrt k).
   - For dummy solution, this means
     - For each k
     - Conduct a cfl search, and record the search cost
     - Record the k corresponds to the minmum search cost
   - For LLM, this means it should both select the k, then, conduct on iterative refinment on cfl on its own.
   - **Note**: This task spans different w selection (w=1,1.5,2), representing slope limiter (ie, 3 k tasks)

```bash
# For example, for profile 1, the 3 tasks are:
python dummy_sols/burgers_1d.py --profile p1 --task k --w 1
python dummy_sols/burgers_1d.py --profile p1 --task k --w 1.5
python dummy_sols/burgers_1d.py --profile p1 --task k --w 2
```

3. **w-Parameter (Composite: ie, 0-shot select w, then iteratively search CFL)**
   - Similar to the k-Parameter task, this is a composite task where w selection is 0-shot, followed by iterative CFL search.
   - For dummy solution, this means:
     - For each w
     - Conduct a cfl search, and record the search cost
     - Record the w corresponds to the minimum search cost
   - For LLM, this means it should select the optimal w value, then conduct iterative refinement on cfl.
   - **Note**: This task spans different k selection (k=-1,0,1), representing different spatial schemes (ie, 3 w tasks)

```bash
# For example, for profile 1, the 3 tasks are:
python dummy_sols/burgers_1d.py --profile p1 --task w --k -1
python dummy_sols/burgers_1d.py --profile p1 --task w --k 0
python dummy_sols/burgers_1d.py --profile p1 --task w --k 1
```

## Summarized parameter table for developer

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| cfl | Courant-Friedrichs-Lewy number for stability | 0 < cfl ≤ 1 |
| k | Blending parameter between central (k=1) and upwind (k=-1) fluxes | -1 ≤ k ≤ 1 |
| w | Parameter for minmod limiter strength | w ≥ 1 |

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| L | Domain length | 2.0 |
| case | Initial condition type | "sin" |
| n_space | Number of grid points | 2048 |
| record_dt | Time interval between recordings | 0.1 |
| end_frame | Simulation end after certain number of frames | 10 |
| dump_dir | Directory for output files | "sim_res/burgers_1d/p1" |
| verbose | Enable verbose output | False |
