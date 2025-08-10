# Solver Checkout Guidelines

This document provides guidelines for solver developers to prepare their solvers for LLM-automation tasks. The checkout process uses a centralized YAML configuration approach to ensure consistent parameter settings, task definitions, and dummy solution generation for downstream LLM development.

## Overview

Each solver should provide approximately **100 individual tasks** distributed across:

- **Variable profiles** (solver-dependent)
- **3 precision levels** (always required)
- **Non-target parameter combinations** (solver-dependent)

## Task Distribution Strategy

### Case A: Solvers with Famous Benchmarks

For solvers with established **benchmark problems where parameters should not be easily changed**:

- **Few profiles** (fixed benchmark setups)
- **More non-target combinations** to reach 100 individuals

```
# eg, for euler 1d, if the targetted parameter is CFL or n_space, 
# we have 9 combos of non-target combinations: (3 choices of k × 3 choices of beta) 
# For 4 target parameters: CFL, n_space (iterative) + k, beta (0-shot)
168 individuals per precision = 1 profile × [(2 iterative × 9 combos) + (2 0-shot × 3 combos)] × 4 target params
```

### Option B: Flexible Parameter Solvers  

For solvers without strict benchmark constraints:

- **More profiles** (can randomize parameters)
- **Fewer non-target combinations** (typically just 1 default)

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
  physics: "[Physics description]"
  reference: "[Literature or standard benchmarks]"
  numerical_method: "[Method description]"

task_distribution:
  strategy: "[benchmark/randomized]"
  total_individuals: [number]
  description: "[calculation breakdown]"

precision_levels:
  high:
    tolerance_linf: [value]
    tolerance_rmse: [value]
    description: "Most stringent convergence criteria"
  # medium and low levels...

profiles:
  strategy: "[benchmark/randomized]"
  active_profiles: ["p1", "p2", ...]
  # profile definitions...

target_parameters:
  [param_name]:
    description: "[parameter description]"
    search_type: "[iterative+0-shot/0-shot]"
    # parameter-specific configuration...
```

### 2. Checkout Document: `[solver_name]_checkout.md`

```markdown
# [Solver Name] Checkout Document

> **Configuration Source**: All parameter settings are centralized in `[solver_name]_config.yaml`. Please refer to that file for the complete and up-to-date configuration details.

## Quick Reference

**Configuration file**: `[solver_name]_config.yaml`  
**Dummy generation script**: `[solver_name]_dummy_generation.py`

## Summary from Configuration

- **Solver**: [Brief description]
- **Method**: [Numerical method]
- **Benchmark**: [Reference problem]
- **Strategy**: [approach with X profiles]
- **Target Parameters**: [number] ([list names])
- **Precision Levels**: [number] ([status])
- **Self-checking**: [list criteria]

## Task Distribution

Current configuration generates:
- **[param types]** (iterative): [calculation]
- **[param types]** (0-shot): [calculation] 
- **Total per precision**: [number] tasks
- **Current active**: [current status]

## Dummy Solution Generation

Refer to the LLM developer script: `[solver_name]_dummy_generation.py`
```

### 3. Generation Script: `[solver_name]_dummy_generation.py`

Python script that reads the YAML configuration and generates all dummy solution tasks.

## Deliverables

1. **`[solver_name]_config.yaml`** - Centralized configuration file with all parameters
2. **`[solver_name]_checkout.md`** - Checkout document referencing the YAML config
3. **`[solver_name]_dummy_generation.py`** - Python script that reads YAML and generates all dummy solutions
4. **Cached dummy solution results** - Run the generation scripts and zip result files; then send to Yadi or Leo

## Implementation Notes

- Use simple parameter names with arrays for multiple choices (e.g., `beta: [1.0, 1.5, 2.0]`)
- Maintain consistency between YAML configuration and generation script logic
- Test YAML loading with `load_config()` and `build_target_configs()` functions
- Verify task count calculations match expected totals
