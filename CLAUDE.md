# CLAUDE.md

This file provides guidance to Claude Code when working with the costsci-tools repository.

## Python Environment

**IMPORTANT**: Always use the `simulcost` conda environment for Python scripts:
```bash
# Activate before running any Python
conda activate simulcost

# Or use full path
/home/ubuntu/miniconda3/envs/simulcost/bin/python script.py
```

## Project Overview

**CostSci-Tools** is the solver library and API for the SimulCost benchmark. It contains 12 physics-based PDE solvers with cost tracking, used to evaluate LLM agents' parameter optimization capabilities.

## Repository Structure

```
costsci-tools/
├── solvers/          # Core numerical implementations (inherit from SIMULATOR base class)
├── runners/          # CLI entry points using Hydra config
├── wrappers/         # High-level APIs: run_sim_*, get_res_*, compare_res_*
├── dummy_sols/       # Brute-force parameter search scripts
├── dataset/          # Per-solver task configurations and results
│   └── {solver}/     # successful/ and failed/ task folders
├── docs/             # Per-solver documentation (physics, parameters, metrics)
├── gen_cfgs/         # Configuration generation scripts
└── checkouts/        # Solver checkout utilities
```

## Solvers (12 total)

| Solver | Description | Key Parameters |
|--------|-------------|----------------|
| burgers_1d | 1D inviscid Burgers (Roe + minmod) | cfl, n_space |
| diff_react_1d | 1D diffusion-reaction | dt, n_space, tol |
| epoch_1d | 1D plasma (EPOCH PIC code) | npart, dt_multiplier |
| euler_1d | 1D compressible Euler | cfl, n_space, beta |
| euler_2d | 2D compressible Euler | cfl, nx |
| fem2d | 2D FEM elasticity | resolution, cg_tolerance |
| hasegawa_mima_linear | Linear Hasegawa-Mima turbulence | N, dt |
| hasegawa_mima_nonlinear | Nonlinear Hasegawa-Mima turbulence | N, dt |
| heat_1d | 1D transient heat conduction | cfl, n_space |
| heat_steady_2d | 2D steady heat (iterative) | dx, error_threshold |
| ns_transient_2d | 2D transient Navier-Stokes | resolution, dt |
| unstruct_mpm | Unstructured MPM solid mechanics | resolution, dt |

## Key Concepts

**Solver Architecture**:
- `solvers/base_solver.py` - SIMULATOR base class with cost tracking
- Each solver tracks `wall_time_total` and sets `wall_time_exceeded` if > 120s
- Cost formulas based on computational complexity (not wall time, except EPOCH)

**Wrapper API Pattern**:
```python
run_sim_{solver}(params, config)     # Execute simulation
get_res_{solver}(params, config)     # Load cached results
compare_res_{solver}(res1, res2)     # Compare solutions (RMSE, L∞)
```

**Task Configuration**:
- Hydra YAML configs in `dataset/{solver}/`
- `successful/` - tasks where brute-force found reference solution
- `failed/` - tasks where search didn't converge

## Documentation

Per-solver docs in `docs/`:
- Physics background and governing equations
- Tunable parameters with ranges
- Cost formula derivation
- Accuracy metrics and thresholds (low/medium/high)

## Development

See `guideline_solver_dev.md` for adding new solvers:
1. Implement solver in `solvers/` (inherit SIMULATOR)
2. Create runner in `runners/` (Hydra CLI)
3. Create wrapper in `wrappers/` (run/get/compare API)
4. Add documentation in `docs/`
5. Generate task configs in `gen_cfgs/`

## Benchmark Statistics

The `benchmark_stats/` folder contains scripts for extracting and verifying benchmark facts:

```bash
# Count tasks per solver
/home/ubuntu/miniconda3/envs/simulcost/bin/python benchmark_stats/count_tasks.py

# Extract precision thresholds
/home/ubuntu/miniconda3/envs/simulcost/bin/python benchmark_stats/extract_thresholds.py
```

**Outputs**:
- `benchmark_stats/task_counts.csv` - Per-solver task counts
- `benchmark_stats/thresholds.csv` - Per-solver precision thresholds

## Quick Commands

```bash
# Run a solver with config
python runners/heat_1d.py +config_path=dataset/heat_1d/successful/task_001

# Run brute-force search
python dummy_sols/heat_1d.py

# Check solver documentation
cat docs/heat_1d.md
```
