# Steady 2D Navier–Stokes — Channel Flow (SIMPLE, Staggered FVM)

This document describes the **steady, incompressible** 2D Navier–Stokes channel-flow solver that uses the **SIMPLE** algorithm on a **staggered finite-volume grid**. It also documents the **task-driven** tuning interface exposed by the CLI.

---

## 1) Problem statement

We solve, in a rectangular domain of length `L` and breadth `B`,

- **Momentum (component-wise)**  
  \[\rho(\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \mu\nabla^2\mathbf{u}\]

- **Continuity**  
  \[\nabla\cdot\mathbf{u} = 0\]

### Discretization & coupling
- **Grid**: staggered. Pressure `p` is stored at cell centers; `u` and `v` at faces. Viscous terms use second‑order central differences; convection uses upwind/central per solver internals.  
- **Pressure–velocity coupling**: **SIMPLE** with under‑relaxation on `u`, `v`, and `p`. The pressure correction step enforces mass conservation.  
- **Convergence**: outer loop stops when (i) mass residual is below `mass_tolerance` and (ii) velocity/pressure RMSE drop below their tolerances (see §5).

---

## 2) Boundary conditions & profiles

Pick a **profile** to configure geometry and BCs:

- `channel_flow` (typical): no‑slip walls (`u=v=0`), prescribed inlet profile (often parabolic or uniform as set in the profile), and a fixed/zero‑gradient outlet pressure.  
- Other geometries present in the repository may include `back_stair_flow`, `expansion_channel`, `cube_driven_flow`. Use the one that matches your experiment; the exact geometry/BC wiring lives with the profile.

You select the profile via `--profile` (default: `p1`) and, if applicable, set `--length` and `--breadth` (defaults below).

---

## 3) CLI overview

The solver exposes a **task** interface to tune one parameter at a time via grid search or simple heuristics. The same executable also supports running a single simulation at chosen values.

### Required
- `--task {mesh_x|mesh_y|omega_u|omega_v|omega_p|diff_u_threshold|diff_v_threshold|res_iter_v_threshold}`

### Common geometry/flow
- `--profile str` (default `p1`)  
- `--length float` (default `20.0`) — channel length  
- `--breadth float` (default `1.0`) — channel height

### Discretization & relaxation (with defaults from `ns_channel_2d.py`)
- `--mesh_x int` (default `250`) — number of cells in x  
- `--mesh_y int` (default `50`) — number of cells in y  
- `--omega_u float` (default `0.7`) — under‑relaxation for `u`  
- `--omega_v float` (default `0.7`) — under‑relaxation for `v`  
- `--omega_p float` (default `0.3`) — under‑relaxation for `p`

### Convergence thresholds
- `--diff_u_threshold float` (default `1e-7`) — per‑iteration update threshold for `u`  
- `--diff_v_threshold float` (default `1e-7`) — per‑iteration update threshold for `v`  
- `--res_iter_v_threshold {float|keyword}` (default `"exp_decay"`) — schedule for the pressure/velocity residual gate. Accepts a **number** (constant threshold) or the keywords **`exp_decay`**, **`linear_decay`** (parsed by `float_or_str`).

### Global stopping & iteration budget
- `--mass_tolerance float` (default `1e-4`) — mass residual target  
- `--u_rmse_tolerance float` (default `3e-2`)  
- `--v_rmse_tolerance float` (default `3e-2`)  
- `--p_rmse_tolerance float` (default `3e-2`)  
- `--max_iter int` (default `20`) — outer SIMPLE iterations (upper bound)

> **Note**: `run_sim_ns_channel_2d(...)` and grid‑search helpers live in `wrappers/`. This script wires tasks → helper functions and passes the geometry/threshold arguments through.

---

## 4) Tasks and what they optimize

Each **task** performs a small search to minimize solve cost (wall time or iteration count) subject to convergence.

- **`mesh_x` / `mesh_y`**  
  Grid refinement along x or y. Typical sweeps: `mesh_x ∈ {64, 128, 256, 512}`, `mesh_y ∈ {16, 32, 64, 128}` at fixed counterpart. Objective is to reach target residual/RMSE with minimal cost. `mesh_y` especially controls near‑wall resolution.

- **`omega_u`, `omega_v`, `omega_p`**  
  Under‑relaxation factors. Helpers commonly try `0.1, 0.2, …, 1.0` for `omega_u/v` and a narrower band for `omega_p`. Higher values can speed convergence but may destabilize; lower values are safer but slower.

- **`diff_u_threshold` / `diff_v_threshold`**  
  Per‑iteration update thresholds for velocity components. Tight thresholds → accuracy ↑, iterations ↑. These are used by the outer stopping criteria together with mass residuals.

- **`res_iter_v_threshold`**  
  Controls how strictly the algorithm treats residuals over iterations. Passing a **float** keeps it constant; `"exp_decay"` or `"linear_decay"` tightens the gate as iterations proceed.

Internally, *grid_search_* / *find_optimal_* helpers run the simulation via `run_sim_ns_channel_2d(...)` with the current candidate parameter and record cost and convergence status. The best converged candidate is reported.

---

## 5) Convergence & diagnostics

At the end of a run (or at each candidate inside a task search), the following are tracked:

- **Mass conservation**: global mass residual ≤ `mass_tolerance`.  
- **Field stability**: RMSE between successive outer iterations for `u`, `v`, `p` falls below `u_rmse_tolerance`, `v_rmse_tolerance`, `p_rmse_tolerance`.  
- **Cost**: iteration count and/or wall time (as provided by wrapper utilities).  
- **Metadata**: chosen task/parameter, mesh, relaxation factors, and schedule keyword/values.

These values are recorded to the console and, when enabled by wrappers, serialized alongside field outputs.

---

## 6) Output artifacts

Depending on your wrapper configuration:
- Field arrays: `u.npy`, `v.npy`, `p.npy` (or an `.h5` bundle).  
- `meta.json`: geometry (`length`, `breadth`), mesh sizes, chosen parameters for the task, tolerances, convergence flags, iteration counts, and (optionally) timing.  
- Optional figures (centerline velocity, residual history) if plotting is enabled in wrappers.

> **Tip**: Keep runs in separate folders per task/sweep (e.g., `dumps/ns2d/p1/omega_u_0p5/`), so you can compare convergence histories cleanly.

---

## 7) Usage examples

### 7.1 Mesh refinement (x then y)
```bash
# Start with a reasonable y-resolution and refine x
python dummy_sols/ns_channel_2d.py \
  --profile p1 --task mesh_x \
  --mesh_x 128 --mesh_y 32 \
  --length 20 --breadth 1 \
  --mass_tolerance 1e-4 --u_rmse_tolerance 3e-2 --v_rmse_tolerance 3e-2 --p_rmse_tolerance 3e-2

# Then refine y holding x fixed
python dummy_sols/ns_channel_2d.py \
  --profile p1 --task mesh_y \
  --mesh_x 256 --mesh_y 32
```

### 7.2 Relaxation tuning
```bash
# omega_u sweep (helpers usually try 0.1..1.0)
python dummy_sols/ns_channel_2d.py --profile p1 --task omega_u --mesh_x 256 --mesh_y 64

# omega_v sweep
python dummy_sols/ns_channel_2d.py --profile p1 --task omega_v --mesh_x 256 --mesh_y 64

# omega_p sweep
python dummy_sols/ns_channel_2d.py --profile p1 --task omega_p --mesh_x 256 --mesh_y 64
```

### 7.3 Thresholds & schedules
```bash
# Fix thresholds tighter for accuracy
python dummy_sols/ns_channel_2d.py --profile p1 --task diff_u_threshold --diff_u_threshold 1e-8
python dummy_sols/ns_channel_2d.py --profile p1 --task diff_v_threshold --diff_v_threshold 1e-8

# Residual schedule: constant number OR a keyword
python dummy_sols/ns_channel_2d.py --profile p1 --task res_iter_v_threshold --res_iter_v_threshold 5e-3
python dummy_sols/ns_channel_2d.py --profile p1 --task res_iter_v_threshold --res_iter_v_threshold exp_decay
python dummy_sols/ns_channel_2d.py --profile p1 --task res_iter_v_threshold --res_iter_v_threshold linear_decay
```

### 7.4 Full custom run (no search), just “set & solve”
```bash
python dummy_sols/ns_channel_2d.py \
  --profile p1 --task omega_u \
  --mesh_x 256 --mesh_y 64 \
  --omega_u 0.6 --omega_v 0.6 --omega_p 0.3 \
  --diff_u_threshold 1e-7 --diff_v_threshold 1e-7 \
  --res_iter_v_threshold exp_decay \
  --length 20.0 --breadth 1.0 \
  --mass_tolerance 1e-4 --u_rmse_tolerance 3e-2 --v_rmse_tolerance 3e-2 --p_rmse_tolerance 3e-2 \
  --max_iter 20
```

---

## 8) Suggested workflow (quick‑start matrix)

| Task | Primary knob(s) | Typical sweep | Goal |
|---|---|---|---|
| `mesh_x` | `mesh_x` @ fixed `mesh_y` | 64 → 128 → 256 → 512 | Accuracy–cost tradeoff along streamwise |
| `mesh_y` | `mesh_y` @ fixed `mesh_x` | 16 → 32 → 64 → 128 | Resolve near‑wall gradients |
| `omega_u` / `omega_v` | `0.1 … 1.0` | coarse→fine around best | Fewer SIMPLE iterations, stable |
| `omega_p` | `0.2 … 0.8` | coarse→fine | Faster mass‑residual closure |
| `diff_u_threshold` / `diff_v_threshold` | `1e-2 … 1e-8` | tighten until stable | Balance accuracy vs. runtime |
| `res_iter_v_threshold` | float or `exp_decay` / `linear_decay` | pick schedule | Robust yet efficient residual gating |

---

## 9) Troubleshooting

- **Divergence when `omega_*` are large** → reduce the offending relaxation(s), especially `omega_p`.  
- **Mass residual stalls** → tighten `res_iter_v_threshold` or try `exp_decay`; also check `mesh_y` near walls.  
- **RMSEs won’t drop** → ensure `diff_*_threshold` aren’t looser than your target; try lowering `max_iter` only after stability is confirmed.  
- **Out‑of‑bounds interpolation errors** (if using wrappers to compare different meshes) → verify target grid extents and that interpolation points lie within the source grid domain.

---

## 10) Reproducibility checklist

- Record: `profile`, `length`, `breadth`, `mesh_x`, `mesh_y`, `omega_u`, `omega_v`, `omega_p`, `diff_u_threshold`, `diff_v_threshold`, `res_iter_v_threshold`, `mass_tolerance`, `u/v/p_rmse_tolerance`, `max_iter`, solver commit hash, and dump directory.  
- Keep one folder per task/setting.  
- Save `meta.json` for each run.

---

### Defaults (from `ns_channel_2d.py`)

```text
profile: p1
mesh_x: 250
mesh_y: 50
omega_u: 0.7
omega_v: 0.7
omega_p: 0.3
diff_u_threshold: 1e-7
diff_v_threshold: 1e-7
res_iter_v_threshold: "exp_decay"   # or a float, or "linear_decay"
max_iter: 20
length: 20.0
breadth: 1.0
mass_tolerance: 1e-4
u_rmse_tolerance: 3e-2
v_rmse_tolerance: 3e-2
p_rmse_tolerance: 3e-2
```
