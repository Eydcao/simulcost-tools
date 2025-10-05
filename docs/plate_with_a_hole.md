## Plate With a Hole (Plane Strain Elasticity, 4-node Quad FEM)

### Introduction

This benchmark solves 2D plane strain linear elasticity for a quarter plate with a circular hole under uniaxial traction. It is used to assess stress concentration and mesh convergence quality.

- Constitutive (plane strain): σ = D ε, with
  D = E / ((1+ν)(1−2ν)) [[1−ν, ν, 0], [ν, 1−ν, 0], [0, 0, (1−2ν)/2]].
- Elements: 4-node bilinear quads (2×2 Gauss integration)
- Domain: quarter plate of size L×H, hole radius R
- BCs: symmetry on x=0 (ux=0) and y=0 (uy=0); uniform traction (σxx=T) on the right edge; elements inside the hole deactivated

### Error Metric (used for precision and convergence)

- Compute σxx at Gauss points; collect on the vertical centerline (x≈0) and outside the hole.
- Compare to the analytical infinite-plate-with-a-hole solution:
  σxx^ana(y) = T (1 + R^2/(2 y^2) + 3 R^4/(2 y^4)).
- Normalized error written by the solver to meta.json as `error` and used by the checkout and dummy strategies.

### Tunable Parameters

- nx: number of elements in x direction
- ny: number of elements in y direction

Other inputs from the solver config: E, ν, traction T, R, L, H.

### Precision Levels (checkout)

Defined in `checkouts/plate_with_a_hole.yaml` (high-first ordering):
- high: tolerance_error=0.001, individual_error_tolerance=0.05
- medium: tolerance_error=0.01, individual_error_tolerance=0.1
- low: tolerance_error=0.1, individual_error_tolerance=0.2

Interpretation:
- individual_error_tolerance: absolute threshold current error must be below
- tolerance_error: permitted change (difference) between successive refinements

### Dummy Strategy (modeled after Euler)

Implemented in `dummy_sols/plate_with_a_hole.py`:
- Iteratively refine the target parameter (nx or ny) while holding the other fixed (multiplication_factor controls refinement).
- Each step runs the solver (via wrapper) and reads error from `sim_res/.../meta.json`.
- Convergence when both hold:
  1) current error ≤ individual_error_tolerance
  2) error difference between consecutive runs < tolerance_error

Public helpers:
- find_convergent_nx(...)
- find_convergent_ny(...)
- Optional grid-search wrappers find_optimal_nx/ny(...)

### How to Run

1) Direct simulation (Hydra runner):
```bash
python runners/plate_with_a_hole.py --config-name=p1 nx=40 ny=40
```

2) Wrapper (returns process output):
```python
from wrappers.plate_with_a_hole import run_simulation
run_simulation(nx=40, ny=40, verbose=True)
```

3) Dummy search (convergence study):
```bash
python dummy_sols/plate_with_a_hole.py --task=nx --nx=40 --ny=40 \
  --tolerance_error=0.01 --individual_error_tolerance=0.1
```

4) Generate checkout cache and statistics:
```bash
python checkouts/plate_with_a_hole.py
```

### Outputs

Per-run outputs in `sim_res/plate_with_a_hole/{profile}_nx_{NX}_ny_{NY}/`:
- frame_0000.h5/json: displacements, mesh, error, parameters
- meta.json: cost, nx, ny, n_elements, n_nodes, n_dof, error

Checkout outputs in `outputs/statistics/`:
- plate_with_a_hole_statistics.png
- plate_with_a_hole_statistics_summary.txt

Datasets are recorded under:
- `dataset/plate_with_a_hole/successful/tasks.json`
- `dataset/plate_with_a_hole/failed/tasks.json`

### Troubleshooting

- Error above individual_error_tolerance: increase nx/ny or refinement steps; ensure traction and material parameters are correct.
- Very coarse meshes near the hole may deactivate too many elements; start with nx, ny ≥ 20.
- If the wrapper cannot find the runner, inspect `_find_runner_path()` in `wrappers/plate_with_a_hole.py`.

### File Map

- Solver: `solvers/plate_with_a_hole.py`
- Runner: `runners/plate_with_a_hole.py`
- Wrapper: `wrappers/plate_with_a_hole.py`
- Dummy: `dummy_sols/plate_with_a_hole.py`
- Checkout config: `checkouts/plate_with_a_hole.yaml`
- Checkout cache script: `checkouts/plate_with_a_hole.py`
