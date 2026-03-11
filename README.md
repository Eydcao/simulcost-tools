# SimulCost Tools

Solver library and API for the SimulCost benchmark. Contains 12 physics-based PDE solvers with cost tracking, used to evaluate LLM agents' parameter optimization capabilities.

## Quick Start

### Setup

First, follow the environment setup instructions in the [main repository README](../README.md#-environment-setup). Then:

```bash
# Activate the simulcost conda environment
conda activate simulcost

# Install costsci_tools in development mode
pip install -e .
```

### Running a Solver

```bash
# Run a solver with default config
python runners/heat_1d.py

# Run with specific config
python runners/heat_1d.py +config_path=dataset/heat_1d/successful/task_001

# Run brute-force parameter search
python dummy_sols/heat_1d.py
```

## Directory Structure

- **solvers/** - Core numerical implementations (inherit from SIMULATOR base class)
- **runners/** - CLI entry points using Hydra configuration framework
- **wrappers/** - High-level API functions: `run_sim_*`, `get_res_*`, `compare_res_*`
- **dummy_sols/** - Brute-force parameter search scripts
- **dataset/** - Per-solver task configurations and results
- **docs/** - Per-solver physics documentation and parameter guides
- **gen_cfgs/** - Configuration generation scripts
- **checkouts/** - Solver checkout/validation utilities
- **run_configs/** - Hydra configuration files for each solver

## Available Solvers

| Solver | Description | Key Tunable Parameters |
|--------|-------------|------------------------|
| burgers_1d | 1D inviscid Burgers equation | `cfl`, `n_space` |
| diff_react_1d | 1D diffusion-reaction | `dt`, `n_space`, `tol` |
| epoch_1d | 1D plasma PIC code | `npart`, `dt_multiplier` |
| euler_1d | 1D compressible Euler | `cfl`, `n_space`, `beta` |
| euler_2d | 2D compressible Euler | `cfl`, `nx` |
| fem2d | 2D FEM elasticity | `resolution`, `cg_tolerance` |
| hasegawa_mima_linear | Linear Hasegawa-Mima turbulence | `N`, `dt` |
| hasegawa_mima_nonlinear | Nonlinear Hasegawa-Mima turbulence | `N`, `dt` |
| heat_1d | 1D transient heat conduction | `cfl`, `n_space` |
| heat_steady_2d | 2D steady heat (iterative) | `dx`, `error_threshold` |
| ns_transient_2d | 2D transient Navier-Stokes | `resolution`, `dt` |
| unstruct_mpm | Unstructured MPM solid mechanics | `resolution`, `dt` |

## Documentation

- **CLAUDE.md** - Developer guide with project overview and quick commands
- **guideline_solver_dev.md** - Instructions for adding new solvers
- **guideline_solver_checkout.md** - Solver validation and checkout guidelines
- **docs/\*.md** - Per-solver physics and parameter documentation

## Architecture

All solvers inherit from the `SIMULATOR` base class in `solvers/base_solver.py`, which provides:
- Unified parameter interface
- Cost tracking (wall time and computational metrics)
- Solution caching and comparison
- Accuracy metrics (RMSE, L∞ norm)

See `docs/` for solver-specific physics background, parameter ranges, and accuracy thresholds.
