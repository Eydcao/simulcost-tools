## Compaction (Plane Strain Elasticity, 4-node Quad FEM)

### Introduction

This benchmark solves 2D plane strain linear elasticity for a bar under self-weight (gravity loading). It is used to assess mesh convergence quality for stress distribution in structural mechanics problems.

- Constitutive (plane strain): σ = D ε, with
  D = E / ((1+ν)(1−2ν)) [[1−ν, ν, 0], [ν, 1−ν, 0], [0, 0, (1−2ν)/2]].
- Elements: 4-node bilinear quads (2×2 Gauss integration)
- Domain: rectangular bar of size length×height
- BCs: bottom edge fixed (ux=uy=0); gravity body force applied throughout domain
- Loading: Self-weight (body force = ρg in y-direction)

### Error Metric (used for precision and convergence)

- Compute σyy (vertical stress) at Gauss points throughout the domain
- Compare to analytical solution for bar under self-weight:
  σyy_analytical(y) = ρg(y - height)
- Normalized error calculated as:
  error = sqrt(Σ(σyy_FEM - σyy_analytical)²) * V / (ρg * height * V * N_gauss)
- Error written by the solver to meta.json as `error` and used by the checkout and dummy strategies

### Tunable Parameters

- nx: number of elements in x direction (typically small, e.g., 2)
- ny: number of elements in y direction (main refinement parameter)

Other inputs from the solver config: E, ν, ρ (density), g (gravity), length, height.

### Precision Levels (checkout)

Defined in `checkouts/compaction.yaml` (high-first ordering):
- high: tolerance_error=2e-4, individual_error_tolerance=2e-4
- medium: tolerance_error=1e-3, individual_error_tolerance=1e-3
- low: tolerance_error=3e-3, individual_error_tolerance=3e-3

Interpretation:
- individual_error_tolerance: absolute threshold current error must be below
- tolerance_error: permitted change (difference) between successive refinements

### Dummy Strategy (modeled after Plate with a Hole)

Implemented in `dummy_sols/compaction.py`:
- Iteratively refine the target parameter (nx or ny) while holding the other fixed (multiplication_factor controls refinement)
- Each step runs the solver (via wrapper) and reads error from `sim_res/.../meta.json`
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
python runners/compaction.py --config-name=p1 nx=2 ny=20
```

2) Wrapper (returns process output):
```python
from wrappers.compaction import run_simulation
run_simulation(nx=2, ny=20, verbose=True)
```

3) Dummy search (convergence study):
```bash
python dummy_sols/compaction.py --function=find_convergent_ny --nx=2 --ny=10 \
  --tolerance_error=1e-3 --individual_error_tolerance=1e-3
```

4) Generate checkout cache and statistics:
```bash
python checkouts/compaction.py
```

5) Visualization:
```bash
python visualize_compaction.py --data_dir sim_res/compaction/p1_nx_2_ny_10
```

### Outputs

Per-run outputs in `sim_res/compaction/{profile}_nx_{NX}_ny_{NY}/`:
- frame_0000.h5/json: displacements, mesh, error, parameters
- meta.json: cost, nx, ny, n_elements, n_nodes, n_dof, error

Checkout outputs in `outputs/statistics/`:
- compaction_statistics.png
- compaction_statistics_summary.txt

Datasets are recorded under:
- `dataset/compaction/successful/tasks.json`
- `dataset/compaction/failed/tasks.json`

### Physical Interpretation

The compaction problem represents a simple structural mechanics benchmark:
- A rectangular bar (e.g., soil column, concrete beam) under its own weight
- Bottom boundary is fixed (foundation support)
- Gravity causes downward displacement and compressive stresses
- Stress increases linearly with depth (σyy = ρg(y - height))
- Maximum displacement occurs at the top of the bar

### Typical Results

For a 1m × 1m bar with E=1e5 Pa, ν=0.0, ρ=1000 kg/m³, g=10 m/s²:
- Coarse mesh (nx=2, ny=10): error ~3e-3
- Medium mesh (nx=2, ny=20): error ~8e-4
- Fine mesh (nx=2, ny=40): error ~2e-4
- Maximum displacement: ~0.05m (5% of height)
- Stress range: 0 to -10000 Pa (compression)
