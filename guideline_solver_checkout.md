# Solver Checkout Guidelines

This document provides guidelines for solver developers to prepare their solvers for LLM-automation tasks. The checkout process uses a centralized YAML configuration approach to ensure consistent parameter settings, task definitions, and dummy solution generation for downstream LLM development.

## Overview

Each solver should provide approximately **100~1000 individual tasks** distributed across:

- **Different profiles** (solver-dependent)
- **3 precision levels** (always required)
- **Non-target parameter combinations** (solver-dependent)

## Task Distribution Strategy

### Option A: Solvers with Famous Benchmarks

For solvers with established **benchmark problems where parameters should not be easily changed**:

- **Few profiles** (fixed benchmark setups)
- **More non-target combinations**, combined with profile number * 3 precision levels, to reach 100~1000 individuals

```
# eg, for euler 1d, if the targetted parameter is CFL or n_space, 
# assume 7 profiles
# we can have 9 combos of non-target combinations: (3 choices of k × 3 choices of beta) 
# if the targetted parameter is k (or beta), 
# we can have fixed CFL and n_space, and 3 choices of beta (or k).
# - **CFL + n_space** (iterative): 7 profiles × 9 non-target combos × 2 target params = 126 tasks
# - **k + beta** (0-shot): 7 profiles × 3 non-target combos × 2 target params = 42 tasks
# - **Total per precision**: 168 tasks
# - **Total tasks**: 504 tasks
```

### Option B: Flexible Parameter Solvers  

For solvers without strict benchmark constraints:

- **More profiles** (can randomize parameters)
- **Fewer non-target combinations** (typically just 1 fixed default value)

```
# eg, for heat 1d, there is not much settings for numerical solver, 
# hence we can generate more initial or boundary conditions as more profiles
100 individuals ≈ 3 precision levels × 30 profiles
```

## Precision Levels

Solver developers must define **3 precision levels** for convergence tolerances:

### Example (Euler 1D)

```yaml
precision_levels:
  high:
    tolerance_linf: 0.2
    tolerance_rmse: 0.02
    description: "Most stringent convergence criteria"
  medium:
    tolerance_linf: [placeholder]
    tolerance_rmse: [placeholder]  
    description: "Moderate convergence criteria"
  low:
    tolerance_linf: [placeholder]
    tolerance_rmse: [placeholder]
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
    initial_value: 1.0
    multiplication_factor: 0.5
    max_iteration_num: 7
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
    search_range: [-1.0, 1.0]
    search_range_slice_num: 11
    non_target_parameters:
      cfl: 0.25       # Fixed stable CFL for most parameter combinations
      beta: [1.0, 1.5, 2.0]  # Array for multiple choices
      n_space: 256    # Fixed moderate resolution for efficiency
```

## Checkout Template

Create the following three files for each solver:

### 1. Configuration File: `[solver_name]_config.yaml`

Centralized configuration containing all parameters:

```yaml
solver_info:
  name: "[Solver Name]"
  numerical_method: "[Method description]"
  reference: "[Literature or standard benchmarks]"

task_distribution:
  strategy: "[benchmark/randomized]"
  total_individuals: [number]
  description: "[calculation breakdown]"

precision_levels:
  high:
    tolerance_linf: [value]
    tolerance_rmse: [value]
  # medium and low levels...

profiles:
  num_profiles: 1
  active_profiles: ["p1"]  # ... add more if needed

target_parameters:
  # Template for iterative+0-shot parameters (e.g., CFL, n_space)
  [iterative_param_name]:
    description: "[parameter description - what it controls]"
    search_type: "iterative+0-shot"
    initial_value: [starting_value]  # Starting point for iterative search
    multiplication_factor: [factor]  # Factor to multiply/divide parameter (e.g., 0.5 for CFL, 2.0 for n_space)
    max_iteration_num: [max_iter]    # Maximum iterations before giving up
    non_target_parameters:
      [fixed_param]: [value]         # Fixed parameter values
      [varying_param]: [value1, value2, value3]  # Arrays for parameter combinations

  # Template for 0-shot parameters (e.g., k, beta)
  [zeroshot_param_name]:
    description: "[parameter description - what it controls]"
    search_type: "0-shot"
    search_range: [min_value, max_value]  # Range for grid search
    search_range_slice_num: [num_slices]  # Number of values to test in range
    non_target_parameters:
      [fixed_param]: [value]         # Fixed parameter values
      [varying_param]: [value1, value2, value3]  # Arrays for parameter combinations
```

### 2. Checkout Document: `[solver_name]_checkout.md`

```markdown
# [Solver Name] Checkout Document

## Summary

- **Solver**: [Physics description (e.g., 1D Euler equations for compressible inviscid flow)]
- **Method**: [Numerical method description]
- **Strategy**: [Approach - benchmark/randomized with X profiles]
- **Benchmark**: [Reference problems (e.g., Sod shock tube problem)]
- **Target Parameters**: [number] ([parameter names])
- **Precision Levels**: [number] ([status - which are defined])

## Task Distribution

Current configuration generates:

- **[iterative params]** (iterative): [X profiles] × [Y non-target combos] × [Z target params] = [total] tasks
- **[0-shot params]** (0-shot): [X profiles] × [Y non-target combos] × [Z target params] = [total] tasks
- **Total per precision**: [total] tasks
- **Total tasks**: 3*[total] tasks

## Configuration file

Refer to the **Configuration file**: `[solver_name]_config.yaml`

## Dummy Solution Generation

Refer to the script: `checkouts/[solver_name]_dummy_generation.py`
```

### 3. Generation Script: `[solver_name]_dummy_generation.py`

Python script that reads the YAML configuration and generates all dummy solution tasks.

## Deliverables

1. **`[solver_name]_checkout.md`** - Checkout document referencing the YAML config and dummy script
2. **`[solver_name]_config.yaml`** - Centralized configuration file with all parameters
3. **`[solver_name]_dummy_generation.py`** - Python script that reads YAML and generates all dummy solutions
4. **Cached dummy solution results** - Run the generation scripts and zip result files; then send to Yadi or Leo
