# Steady State Navier-Stokes Simulation in 2D Channel Flow

This document describes the 2D steady-state Navier-Stokes (NS) simulation using the SIMPLE algorithm in a rectangular channel. The case models incompressible flow under varying boundary conditions and supports task-driven parameter tuning.

---

## Problem Description

We solve the incompressible, steady-state Navier-Stokes equations in 2D using Finite Volume Method (FVM) on a staggered grid. Pressure-velocity coupling is handled via the SIMPLE algorithm.

**Governing Equations:**
- Momentum (u and v): \[ \rho (u \cdot \nabla u) = -\nabla p + \mu \nabla^2 u \]
- Continuity: \[ \nabla \cdot u = 0 \]

**Boundary conditions:**

- `channel_flow`: u = 1 at the center of inlet (parabolic), u = 0 at walls, pressure fixed at outlet.
- `back_stair_flow`, `expansion_channel`, and `cube_driven_flow` also supported (custom geometry).

---

## Tunable Parameters

| Parameter             | Description                                              |
|----------------------|----------------------------------------------------------|
| mesh_x, mesh_y       | Grid resolution in x and y                               |
| omega_u, omega_v     | Relaxation factors for u and v velocity correction       |
| omega_p              | Relaxation factor for pressure correction                |
| diff_u_threshold     | Threshold for u-velocity update residual                 |
| diff_v_threshold     | Threshold for v-velocity update residual                 |
| res_iter_v_threshold | Residual norm scheduler for pressure correction          |

---

## Dummy Solution Tuning Tasks

### 1. Finding Optimal Grid Resolution (Task: `mesh_x/mesh_y`)

Finer mesh improves accuracy but increases computational cost. Dummy method starts with coarse grid and refines until centerline velocity profile converges.

```bash
python dummy_sols/ns_channel_2d.py --task mesh_x --profile p1 --initial_mesh_x 50 --initial_mesh_y 10
python dummy_sols/ns_channel_2d.py --task mesh_y --profile p1 --initial_mesh_x 50 --initial_mesh_y 10
```

---

### 2. Finding Optimal Relaxation Factors (Task: `omega_u/omega_v/omega_p`)

Optimal SOR factors reduce iterations. Dummy method uses grid search between 0.1 and 1.9 (step 0.1) for `omega_u`, `omega_v`, and `omega_p`.

```bash
python dummy_sols/ns_channel_2d.py --task omega_u --profile p1
python dummy_sols/ns_channel_2d.py --task omega_v --profile p1
python dummy_sols/ns_channel_2d.py --task omega_p --profile p1
```

---

### 3. Finding Optimal Velocity Residual Thresholds (Task: `diff_u_threshold/diff_v_threshold`)

These affect convergence criteria. Dummy method decays `diff_u_threshold` and `diff_v_threshold` iteratively and observes stability and runtime.

```bash
python dummy_sols/ns_channel_2d.py --task diff_u_threshold --profile p1 --initial_threshold 1e-3
python dummy_sols/ns_channel_2d.py --task diff_v_threshold --profile p1 --initial_threshold 1e-3
```

---

### 4. Finding Optimal Pressure Residual Scheduler (Task: `res_iter_v_threshold`)

This can be a constant or a decay function like `exp_decay`. Dummy method tests different scheduling strategies and selects the one minimizing mass residual.

```bash
python dummy_sols/ns_channel_2d.py --task res_iter_v_threshold --profile p1
```

---

## Output

All results are saved to the specified `dump_dir`, including:
- `{filebase}.h5` (`u.npy`, `v.npy`, `p.npy`): final field results
- `meta.json`: convergence metadata (e.g. mass conservation status, number of iterations, runtime, etc.)

---

## Config Generation for Test Cases

Randomize environment parameters (`length`, `breadth`, `mu`, `rho`) to generate new test cases:

```bash
python gen_cfgs/ns_channel_2d.py
```

This will output new YAML config files in `configs/ns_channel_2d/` folder for varied scenarios, keeping solver robustness and reproducibility in check.

---

## Available Profiles:
- `p1` (default used in experiments)
- `p2/p3/p4/p5`

Each profile applies different geometry and flow constraints. Ensure correct `boundary_condition` is used in the config.

---

End of documentation.
