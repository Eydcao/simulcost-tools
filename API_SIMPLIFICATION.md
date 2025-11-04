# Wall Time API Simplification

## Date: 2025-11-04

## Overview

Simplified the wall time API to be cleaner and easier to understand. Removed unnecessary complexity.

## Changes Made

### 1. Simplified max_wall_time Parameter

**Previous behavior** (confusing):
```python
get_results(profile, N, dt, max_wall_time=None)
# max_wall_time=None: Use config default
# max_wall_time=240: Override to 240 seconds
# max_wall_time=-1: Disable limit
```

**New behavior** (simple):
```python
get_results(profile, N, dt, max_wall_time=-1)
# max_wall_time=-1 (default): Use config default
# max_wall_time=None: Disable limit (no constraint)
# max_wall_time=240: Override to 240 seconds
```

**Benefits**:
- Clearer semantics: `None` naturally means "no limit"
- Default value (-1) explicitly means "use config default"
- Easier to understand at call sites

### 2. Removed max_wall_time_multiplier from compare_solutions()

**Previous behavior** (complex):
```python
compare_solutions(profile, params1, params2, tolerance_rmse,
                 max_wall_time_multiplier=2.0)
# Multiplier applies to second run
# max_wall_time_multiplier=0: Disable limit for second run
```

**New behavior** (simple):
```python
compare_solutions(profile, params1, params2, tolerance_rmse)
# Second run ALWAYS has no wall time limit
# Allows higher resolution simulations to complete
```

**Benefits**:
- One less parameter to think about
- Second run will never have false rejections due to wall time
- Simpler implementation - no complex multiplier logic

## Updated APIs

### get_results() - Nonlinear Wrapper

```python
def get_results(profile, N, dt, max_wall_time=-1):
    """
    Args:
        max_wall_time: Override for maximum wall time in seconds (default=-1).
                      - Default (-1): Use config default (120s for nonlinear)
                      - None: Disable limit (no constraint)
                      - Positive number: Override to that many seconds

    Returns:
        (cost, results_list, simulation_completed)
    """
```

**Examples**:
```python
# Use config default (120s)
cost, results, completed = get_results('p1', N=128, dt=5.0)

# Disable wall time limit (no constraint)
cost, results, completed = get_results('p1', N=256, dt=2.5, max_wall_time=None)

# Override to 300 seconds
cost, results, completed = get_results('p1', N=256, dt=2.5, max_wall_time=300)
```

### run_sim_hasegawa_mima_linear() - Linear Wrapper

```python
def run_sim_hasegawa_mima_linear(profile, N, dt, cg_atol, analytical, max_wall_time=-1):
    """
    Args:
        max_wall_time: Override for maximum wall time in seconds (default=-1).
                      - Default (-1): Use config default (60s for linear)
                      - None: Disable limit (no constraint)
                      - Positive number: Override to that many seconds

    Returns:
        cost
    """
```

**Examples**:
```python
# Use config default (60s)
cost = run_sim_hasegawa_mima_linear('p1', N=128, dt=5.0, cg_atol=1e-8, analytical=False)

# Disable wall time limit
cost = run_sim_hasegawa_mima_linear('p1', N=256, dt=2.5, cg_atol=1e-10, analytical=False, max_wall_time=None)

# Override to 120 seconds
cost = run_sim_hasegawa_mima_linear('p1', N=256, dt=2.5, cg_atol=1e-10, analytical=False, max_wall_time=120)
```

### compare_solutions() - Nonlinear Wrapper

```python
def compare_solutions(profile, params1, params2, tolerance_rmse):
    """
    Compare two simulations. First run uses config default for wall time.
    Second run has NO wall time limit (allows higher resolutions to complete).

    Args:
        profile: Configuration profile name
        params1: Dictionary with first simulation parameters (coarse)
        params2: Dictionary with second simulation parameters (fine)
        tolerance_rmse: RMSE tolerance for convergence

    Returns:
        (is_converged, cost1, cost2, rmse_diff)
    """
```

**Behavior**:
- First run: Uses config default wall time (120s for nonlinear)
- If first run hits wall time limit → Skip second run (early termination)
- Second run: NO wall time limit (max_wall_time=None)

**Example**:
```python
# Compare N=64 vs N=128
is_converged, cost1, cost2, rmse = compare_solutions(
    'p1',
    {'N': 64, 'dt': 5.0},
    {'N': 128, 'dt': 5.0},
    tolerance_rmse=0.0005
)
# First run (N=64): Uses 120s limit
# Second run (N=128): NO limit - will complete regardless of time
```

## Implementation Details

### Wrappers

Both `hasegawa_mima_nonlinear.py` and `hasegawa_mima_linear.py` use the same logic:

```python
# Add max_wall_time override if specified
if max_wall_time is None:
    # Disable wall time limit
    cmd += " max_wall_time=null"
elif max_wall_time > 0:
    # Override to specific value
    cmd += f" max_wall_time={max_wall_time}"
# If max_wall_time == -1 (default), don't add anything - use config default
```

### compare_solutions

```python
# Run first (coarse) simulation with normal wall time limit (use config default)
cost1, results1, completed1 = get_results(profile, **params1)

# Early termination if first failed
if not completed1:
    print(f"⚠️  First simulation hit wall time limit - skipping second simulation")
    return False, cost1, 0, None

# Run second (fine) simulation with NO wall time limit
# This allows higher resolution simulations to complete without false rejections
cost2, results2, completed2 = get_results(profile, max_wall_time=None, **params2)
```

## Migration from Old API

No code changes needed in dummy_sols or callers! The default behavior remains the same:

**Old code** (still works):
```python
cost, results, completed = get_results('p1', N=128, dt=5.0)
# Used max_wall_time=None (default) → Used config default
```

**New code** (equivalent):
```python
cost, results, completed = get_results('p1', N=128, dt=5.0)
# Uses max_wall_time=-1 (new default) → Uses config default
```

## Testing

✅ Tested with direct function call:
```bash
$ python -c "from dummy_sols.hasegawa_mima_nonlinear import find_convergent_N; ..."
Running new nonlinear simulation with parameters: N=32, dt=10.0, max_wall_time=-1
✅ Simplified API test passed: True 64
```

Shows that default behavior (`max_wall_time=-1`) correctly uses config default.

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Default parameter | `max_wall_time=None` | `max_wall_time=-1` |
| Disable limit | `max_wall_time=-1` | `max_wall_time=None` |
| Use config default | `max_wall_time=None` | `max_wall_time=-1` (default) |
| compare_solutions params | 4 params (with multiplier) | 3 params (no multiplier) |
| Second run wall time | Multiplier-based | Always disabled |

**Result**: Simpler, clearer API that's easier to understand and use.
