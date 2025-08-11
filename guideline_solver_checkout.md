# Solver Checkout Guidelines

This document provides guidelines for solver developers to prepare their solvers for LLM-automation tasks. The checkout process uses delivers a config file and a section in the main document to ensure consistent parameter and task definitions, and dummy solution generation.

## Task Distribution Strategy

Each **tunnable parameter of the solver** should provide approximately **30~100 individual variations (tasks)**; Hence each solver should provide about **200~1000 individual tasks**. This can be distributed across:

- **Different profiles** (solver-dependent)
- **3 precision levels** (always required)
- **Non-target parameter combinations** (solver-dependent)

### Strategy A: Famous Benchmark

For solvers with famous benchmark problems:

- **Few profiles** (3-5 fixed benchmark setups)
- **More non-target combinations** to reach target task count

**Example (Euler 1D)**:

```
# eg, for euler 1d, if the targetted parameter is CFL or n_space, 
# assume 3 profiles (sod, lax, mach_3)
# we have 9 combos of non-target combinations: (3 choices of k × 3 choices of beta) 
# if the targetted parameter is k (or beta), 
# we have fixed CFL and n_space, and 3 choices of beta (or k).

- CFL/n_space (iterative): 27 = 3 profiles × 9 non-target combos tasks each  
- k/beta (0-shot): 9 tasks each
- Total per precision: 72 = 27+27+9+9 tasks
- Across 3 precisions: 216 tasks
```

### Strategy B: Flexible Parameter Solvers  

For solvers without famous benchmarks:

- **More profiles** (can randomize parameters)
- **Fewer non-target combinations** (typically 1 default value)

```
# eg, for heat 1d, there is not much settings for numerical solver, 
# hence we can generate more initial or boundary conditions as more profiles
90 variations = 3 precision levels × 30 profiles
```

## Precision Levels

Solver developers must define **3 precision levels** for convergence tolerances:

### Example (Has to be determined/finetuned by solver developers)

```yaml
precision_levels:
  high:
    tolerance_rmse: 0.0025
    description: "Most stringent convergence criteria"
  medium:
    tolerance_rmse: 0.005  
    description: "Moderate convergence criteria"
  low:
    tolerance_rmse: 0.01
    description: "Relaxed convergence criteria"
```

## Parameter Configuration

### Define Both Target and Non-Target Parameters

For each tunable parameter, when it's the a search target, solver developers must define its search type (0-shot or 0-shot+iterative), then depending on the type, developer must define related search range + the other parameters' choices

Eg, for Euler 1d, assume CFL is the target, the others are fixed:

```yaml
target_parameters:
  cfl:
    description: "Determine the CFL number for temporal stability - controls time step size relative to grid spacing"
    search_type: "iterative+0-shot"
    # ...
    non_target_parameters:
      n_space: 256    # Fixed moderate resolution for efficiency
      beta: [1.0, 1.5, 2.0]  # Array for multiple choices
      k: [-1.0, 0.0, 1.0]    # Array for multiple choices
```

assume k is the target

```yaml
target_parameters:
  k:
    description: "Spatial scheme blending parameter (1.0=central, -1.0=upwind, 0.0=mixed)"
    search_type: "0-shot"
    # ...
    non_target_parameters:
      cfl: 0.25       # Fixed stable CFL for most parameter combinations
      beta: [1.0, 1.5, 2.0]  # Array for multiple choices
      n_space: 256    # Fixed moderate resolution for efficiency
```

## Deliverables

1. **Updated `docs/[solver_name].md`** - Main documentation with integrated checkout section
2. **`checkouts/[solver_name]_config.yaml`** - Centralized configuration file with all parameters  
3. **`checkouts/[solver_name]_dummy_generation.py`** - Python script that reads YAML and generates all dummy solutions
4. **Cached dummy solution results** - Run the generation scripts and zip result files; then send to Yadi or Leo

## Template

Please use Euler 1D as an example:

1. `docs/euler_1d.md`
2. `checkouts/euler_1d_config.yaml`
3. `checkouts/euler_1d_dummy_generation.py`
