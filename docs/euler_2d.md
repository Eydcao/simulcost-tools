# Euler 2D Equations with Advection-Projection Method

## Introduction

This simulation solves the 2D compressible Euler equations for inviscid gas dynamics using an advection-projection method with high-order WENO reconstruction and TVD Runge-Kutta time integration:

**Conservative form:**
$$\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = 0$$

Where the conservative variables and flux are:
$$\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ \rho E \end{pmatrix}, \quad \mathbf{F} = \begin{pmatrix} \rho \mathbf{u} \\ \rho u \mathbf{u} + p\mathbf{e}_x \\ \rho v \mathbf{u} + p\mathbf{e}_y \\ \mathbf{u}(\rho E + p) \end{pmatrix}$$

**Primitive variables:**

- $\rho$ = density
- $u, v$ = velocity components in x and y directions
- $p$ = pressure
- $E$ = specific total energy

**Equation of state (ideal gas):**
$$p = (\gamma - 1) \rho \left(E - \frac{u^2 + v^2}{2}\right)$$

where $\gamma = 1.4$ is the ratio of specific heats for air.

### Numerical Method: Advection-Projection

The solver uses a fractional-step method that splits the computation into two stages per time step:

1. **Advection Step**: Update conservative variables via pure advection, as if there is no influence of the pressure
   - High-order WENO reconstruction (3rd order) for spatial discretization
   - Local Lax-Friedrichs Riemann solver for flux computation
   - TVD Runge-Kutta (3rd order) for time integration

2. **Projection Step**: Enforce divergence-free constraint on velocity field
   - Solve pressure Poisson equation: $\nabla^2 p = \text{RHS}$, where RHS is derived to account for the influence of the internal pressure
   - Use Conjugate Gradient (CG) iterative solver

### Spatial Discretization

The spatial discretization uses:

- **Cartesian grid**: Uniform spacing $\Delta x = \Delta y = 1 / N_x$
- **WENO reconstruction**: Weighted Essentially Non-Oscillatory scheme for high accuracy near discontinuities
  - WENO3: 3rd order accuracy
- **Riemann solver**: Local Lax-Friedrichs flux at cell interfaces
- **Ghost layers**: 2-cell boundary layers for accurate boundary conditions

### Temporal Discretization

Time integration uses TVD Runge-Kutta schemes:

- **TVDRK3**: 3rd order, 3 substeps per time step

**Adaptive time stepping** with CFL condition:
$$\Delta t = \text{CFL} \cdot \frac{2 \Delta x}{|u|+\sqrt{u^2 + 4c^2}}$$

where $c = \sqrt{\gamma p / \rho}$ is the local speed of sound.

Minimum timestep bounds (for extreme shocks):

- Standard: $\Delta t_{\text{min}} = 10^{-10}$
- Strong shocks (testcase 5): $\Delta t_{\text{min}} = 10^{-11}$
- High Mach (testcase 6): $\Delta t_{\text{min}} = 10^{-8}$

### Boundary Conditions

The solver supports four boundary condition types:

- **FREE**: Free boundaries (extrapolation from interior)
- **BOUND**: Wall boundaries (no-slip/slip)
- **INLET**: Prescribed inflow with fixed conservative variables
- **GAS**: Interior gas cells (default)

## Test Cases

The solver provides 9 test cases spanning true 2D flows and pseudo-1D problems. Test cases 0-1 are genuine 2D problems, while cases 2-8 mimic 1D problems using thin 2D grids (aspect ratio = 1/Nx).

### True 2D Problems

**Case 0 (p1): Central Explosion**

- Circular high-pressure region in center of domain
- Aspect ratio: 1.0 (square domain)
- Domain: $[-0.5, 0.5] \times [-0.5, 0.5]$
- Initial conditions:
  - Center (radius $r \leq 0.04N_x$): $\rho=1.0$, $p=2.5$, $u=v=0$
  - Ambient: $\rho=0.125$, $p=0.25$, $u=v=0$
- Boundary: FREE on all sides
- Duration: 10 frames × 0.075s = 0.75s
- Tests: Radial symmetry, circular shock propagation

**Case 1 (p2): Stair Flow**

- Supersonic flow over step geometry
- Aspect ratio: 0.25 (wide domain, 4:1)
- Domain: $[-0.5, 0.5] \times [-0.125, 0.125]$
- Initial conditions:
  - Uniform supersonic flow: $\rho=1.4$, $p=1.0$, $u=3.0$, $v=0$
- Geometry: Step at $x \geq 0.25$, $y < 0.25$ (blocked)
- Boundary: INLET (left), BOUND (top/bottom/step)
- Duration: 10 frames × 0.075s = 0.75s
- Tests: Shock reflection, flow separation, oblique shocks

### Pseudo-1D Problems (Thin 2D Grids)

These test cases use aspect ratio = 1/Nx to create thin grids with ny=2 interior cells, effectively mimicking 1D behavior while using the 2D solver. Top and bottom boundaries are walls (BOUND).

**Case 2 (p3): Sod Shock Tube**

- Classic Riemann problem
- Left ($x < 0$): $\rho=1.0$, $p=2.5$, $u=v=0$
- Right ($x \geq 0$): $\rho=0.125$, $p=0.25$, $u=v=0$
- Boundary: BOUND (top/bottom), FREE (left/right)
- Duration: 10 frames × 0.02s = 0.2s

**Case 3 (p4): Lax Shock Tube**

- Similar to Sod with inlet BC
- Left ($x < 0$): $\rho=0.445$, $E=8.9284$, $\rho u=0.31061$, $\rho v=0$
- Right ($x \geq 0$): $\rho=0.5$, $E=1.425$, $\rho u=0$, $\rho v=0$
- Boundary: INLET (left), BOUND (top/bottom)
- Duration: 10 frames × 0.02s = 0.2s

**Case 4 (p5): Mach 3 Problem**

- High Mach number shock
- Left ($x < 0$): $\rho=3.857$, $E=27.46478$, $\rho u=3.54844$, $\rho v=0$
- Right ($x \geq 0$): $\rho=1.0$, $E=8.80125$, $\rho u=3.55$, $\rho v=0$
- Boundary: INLET (left), FREE (right), BOUND (top/bottom)
- Duration: 10 frames × 0.02s = 0.2s

**Case 5 (p6): Strong Shock Tube**

- Extreme pressure ratio ($2.5 \times 10^{10}$)
- Left ($x < 0$): $\rho=1.0$, $E=2.5 \times 10^{10}$, $\rho u=0$, $\rho v=0$
- Right ($x \geq 0$): $\rho=0.125$, $E=0.25$, $\rho u=0$, $\rho v=0$
- Boundary: BOUND (top/bottom), FREE (left/right)
- Duration: 10 frames × 0.02s = 0.2s
- Special: Minimum timestep $\Delta t_{\text{min}} = 10^{-11}$ for stability

**Case 6 (p7): High Mach Problem**

- Very high velocity shock (Mach $\approx 20000$)
- Left ($x < 0$): $\rho=10.0$, $E=20001250$, $\rho u=20000$, $\rho v=0$
- Right ($x \geq 0$): $\rho=20.0$, $E=1250$, $\rho u=0$, $\rho v=0$
- Boundary: INLET (left), FREE (right), BOUND (top/bottom)
- Duration: 10 frames × 0.02s = 0.2s
- Special: Minimum timestep $\Delta t_{\text{min}} = 10^{-8}$ for stability

**Case 7 (p8): Interacting Blast Shock**

- Two blast waves colliding
- Left 10% ($x < -0.4$): $\rho=1.0$, $E=2500$, $\rho u=0$, $\rho v=0$
- Middle 80%: $\rho=1.0$, $E=0.025$, $\rho u=0$, $\rho v=0$
- Right 10% ($x > 0.4$): $\rho=1.0$, $E=250$, $\rho u=0$, $\rho v=0$
- Boundary: BOUND on all sides (closed tube)
- Duration: 10 frames × 0.02s = 0.2s

**Case 8 (p9): Symmetric Rarefaction Waves**

- Opposite velocity collision creating vacuum
- Left ($x < 0$): $\rho=1.0$, $E=3.0$, $\rho u=-2.0$, $\rho v=0$
- Right ($x \geq 0$): $\rho=1.0$, $E=3.0$, $\rho u=2.0$, $\rho v=0$
- Boundary: INLET (left/right), BOUND (top/bottom)
- Duration: 10 frames × 0.015s = 0.15s

### Convergence Criteria

The simulated results are considered correct if the Normalized RMSE meets the precision-dependent tolerance compared to a reference solution (typically adjacent finer grid or tighter CG tolerance):

$$\text{NRMSE} = \frac{\|\mathbf{f}_1 - \mathbf{f}_2\|_2}{\max(|\mathbf{f}_1|)}$$

The final RMSE is the average of NRMSE over density and pressure fields:
$$\text{RMSE} = \frac{\text{NRMSE}(\rho) + \text{NRMSE}(p)}{2}$$

Convergence thresholds by precision level:

- **Low**: RMSE < 0.08
- **Medium**: RMSE < 0.04
- **High**: RMSE < 0.008

When comparing different grid resolutions, the coarser solution is interpolated to the finer grid before computing NRMSE. Pressure is vertex-centered (at $i \cdot \Delta x$), while density and velocity are cell-centered (at $(i+0.5) \cdot \Delta x$).

## Parameter Tuning Tasks and Dummy Strategy

### Tasks

1. **n_grid_x Convergence Search (iterative+0-shot)**
   - n_grid_x determines spatial resolution: $\Delta x = \Delta y = 1 / n\_grid\_x$
   - Grid spacing affects both accuracy and timestep (via CFL condition)

2. **CFL Convergence Search (iterative+0-shot)**
   - CFL (Courant-Friedrichs-Lewy) number controls timestep: $\Delta t = \text{CFL} \cdot \Delta x / (|u| + c)$
   - Smaller CFL → smaller timesteps → more stable but more expensive

3. **cg_tolerance Optimization (iterative+0-shot)**
   - Convergence tolerance for CG solver in projection step
   - Tighter tolerance → more CG iterations → higher accuracy but more expensive

### Dummy Strategy

1. **n_grid_x Convergence Search (iterative+0-shot)**
   - For dummy solution, double n_grid_x each iteration (multiplication factor: 2) starting from 16 until convergence
   - **Non-target parameters**: cfl=0.25, cg_tolerance=1e-7

2. **CFL Convergence Search (iterative+0-shot)**
   - For dummy solution, halve CFL each iteration (multiplication factor: 0.5) starting from 0.5 until convergence
   - **Non-target parameters**: n_grid_x∈{32, 64, 128, 256}, cg_tolerance=1e-7

3. **cg_tolerance Optimization (iterative+0-shot)**
   - For dummy solution, reduce cg_tolerance by 10× each iteration (multiplication factor: 0.1) starting from 1e-2 until convergence
   - **Non-target parameters**: n_grid_x∈{32, 64, 128, 256}, cfl=0.25

## Summarized parameter table for developer only (Not LLM)

### Controllable

| Parameter | Description | Range |
|-----------|-------------|-------|
| n_grid_x | Number of grid cells in x-direction (y computed from aspect ratio) | 16 ≤ n_grid_x ≤ 512 |
| cfl | Courant-Friedrichs-Lewy number for timestep stability | 0.03125 ≤ cfl ≤ 0.5 |
| cg_tolerance | CG solver convergence tolerance for pressure projection | 1e-6 ≤ cg_tolerance ≤ 1e-2 |

More Notes:

- Smaller n_grid_x → coarser grid → lower accuracy but lower cost
- Larger n_grid_x → finer grid → higher accuracy but scales as $O(N_x^2 \cdot N_{\text{steps}})$
- Grid spacing: $\Delta x = 1 / n\_grid\_x$
- Smaller CFL → more stable but more time steps → higher cost
- Larger CFL → fewer time steps but risk of instability (CFL ≤ 0.5 for stability)
- Tighter cg_tolerance → more CG iterations per projection → higher accuracy but higher cost per step
- Looser cg_tolerance → may cause divergence or loss of mass conservation

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| testcase | Test case selection (0-8) | 0 |
| gamma | Ratio of specific heats | 1.4 |
| start_frame | Starting frame number | 0 |
| end_frame | Number of frames to simulate | 10 |
| record_dt | Time interval between recordings | varies by case |
| dump_dir | Directory for output files | "sim_res/euler_2d/p1" |
| verbose | Enable verbose output | False |

## Cost Analysis

The computational cost is tracked as:

$$\text{Total Cost} = N_{\text{cells}} \times (N_{\text{steps}} + N_{\text{CG\_iters}})$$

Where:

- $N_{\text{cells}} = N_x \times N_y$ = total number of grid cells
- $N_{\text{steps}}$ = total number of advection steps taken
- $N_{\text{CG\_iters}}$ = total CG iterations across all projection steps

This cost metric captures:

1. **Advection Cost**: Each advection step processes all grid cells
   - WENO reconstruction at cell interfaces
   - Riemann solver flux computation
   - TVD-RK substeps (2 or 3 substeps per step)
   - Scales as: $N_{\text{cells}} \times N_{\text{steps}}$

2. **Projection Cost**: Each CG iteration operates on the full grid
   - Sparse matrix-vector multiply for Laplacian operator
   - Vector operations (dot products, saxpy)
   - Typically 5-50 CG iterations per projection step
   - Scales as: $N_{\text{cells}} \times N_{\text{CG\_iters}}$

3. **Time Step Cost**: Number of steps determined by:
   - Simulation duration: $T_{\text{end}} = \text{end\_frame} \times \text{record\_dt}$
   - CFL timestep: $\Delta t = \text{CFL} \cdot \Delta x / (|u|_{\max} + c_{\max})$
   - Total steps: $N_{\text{steps}} \approx T_{\text{end}} / \Delta t$

**Parameter Impact on Cost**:

- **n_grid_x**: Increases cost quadratically ($N_{\text{cells}} \propto N_x^2$) and increases steps linearly ($\Delta t \propto \Delta x$)
  - Total scaling: $\sim O(N_x^3)$
- **cfl**: Decreases cost linearly (larger CFL → fewer steps)
  - Doubling CFL approximately halves the number of steps
- **cg_tolerance**: Affects CG iteration count
  - Tighter tolerance → more CG iterations (roughly logarithmic: reducing by 10× adds ~10-20 iterations)

The total cost provides a comprehensive measure of computational work that accounts for both spatial resolution (via grid cells) and temporal integration (via advection steps and CG iterations).

## Checkout

### Summary

- **Benchmarks**:
  - **p1**: Central explosion - circular blast wave (true 2D)
  - **p2**: Stair flow - supersonic flow over step (true 2D)
  - **p3**: Sod shock tube (pseudo-1D)
  - **p4**: Lax shock tube (pseudo-1D)
  - **p5**: Mach 3 problem (pseudo-1D)
  - **p6**: Strong shock tube (pseudo-1D)
  - **p7**: High Mach problem (pseudo-1D)
  - **p8**: Interacting blast shock (pseudo-1D)
  - **p9**: Symmetric rarefaction waves (pseudo-1D)
- **Target Parameters**: 3 (n_grid_x, cfl, cg_tolerance)
- **Precision Levels**: 3 (low: 0.08, medium: 0.04, high: 0.008)

### Task Distribution

Current configuration generates:

- **n_grid_x** (iterative+0-shot): 9 profiles × 1 non-target combo = 9 tasks
- **cfl** (iterative+0-shot): 9 profiles × 4 non-target combos = 36 tasks
- **cg_tolerance** (iterative+0-shot): 9 profiles × 4 non-target combos = 36 tasks
- **Total per precision**: 81 tasks
- **Total tasks**: 243 tasks (across 3 precision levels)

### Dummy Solution Cache

Config for dummy solution cache: `checkouts/euler_2d.yaml`
Cache script: `checkouts/euler_2d.py`
