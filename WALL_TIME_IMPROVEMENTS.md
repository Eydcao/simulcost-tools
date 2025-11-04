# Wall Time Constraint Improvements

## Overview
Improved wall time handling to be smarter about when to apply limits and when to skip expensive comparisons.

## Key Improvements

### 1. Configurable Wall Time Per Run

**Problem:** Previously, `max_wall_time` was a global config setting. When comparing resolutions (e.g., N=128 vs N=256), both used the same limit, causing false rejections.

**Solution:** `max_wall_time` can now be overridden per simulation run:

```python
# Use config default
get_results(profile, N=128, dt=5.0)

# Override to allow more time
get_results(profile, N=256, dt=5.0, max_wall_time=240)

# Disable limit for comparison
get_results(profile, N=256, dt=5.0, max_wall_time=-1)
```

### 2. Early Termination for Failed Comparisons

**Problem:** If a coarse resolution (N=64) hits the wall time limit, running a finer resolution (N=128) will definitely also fail. Wasted computation.

**Solution:** `compare_solutions()` now checks if the first simulation completed before running the second:

```python
# Run first (coarse) simulation
cost1, results1, completed1 = get_results(profile, **params1)

if not completed1:
    # First simulation hit wall time limit
    # Skip second simulation entirely
    return False, cost1, 0, None

# Only run second simulation if first succeeded
cost2, results2, completed2 = get_results(profile, **params2)
```

### 3. Extended Time for Comparison Runs

**Problem:** Higher resolutions naturally take longer. A fair comparison needs to allow them appropriate time.

**Solution:** `compare_solutions()` accepts `max_wall_time_multiplier`:

```python
# Allow 2x time for the finer resolution
compare_solutions(profile, params1, params2, tolerance,
                 max_wall_time_multiplier=2.0)

# Disable limit entirely for comparison
compare_solutions(profile, params1, params2, tolerance,
                 max_wall_time_multiplier=0)
```

## API Changes

### Nonlinear Wrapper

**`get_results(profile, N, dt, max_wall_time=None)`**
- `max_wall_time=None`: Use config default (120s)
- `max_wall_time=240`: Override to 240 seconds
- `max_wall_time=-1`: Disable limit

**`compare_solutions(profile, params1, params2, tolerance_rmse, max_wall_time_multiplier=2.0)`**
- Early exits if first simulation fails
- Applies multiplier to second simulation's time limit
- `max_wall_time_multiplier=0`: Disables limit for second run

### Linear Wrapper

**`run_sim_hasegawa_mima_linear(profile, N, dt, cg_atol, analytical, max_wall_time=None)`**
- Same behavior as nonlinear `get_results()`

## Benefits

1. **Avoids False Rejections:** Higher resolutions get appropriate time
2. **Saves Computation:** Skips doomed-to-fail fine resolution runs
3. **More Flexible:** Can disable limits when needed for validation
4. **Smarter Search:** Dummy solutions don't waste time on hopeless cases

## Example: Resolution Convergence

**Old Behavior:**
```
N=64, dt=2.5:  Completed in 40s ✅
N=128, dt=2.5: Hit 120s limit ❌ (needed 160s)
Result: False rejection - N=128 might have converged if given time
```

**New Behavior:**
```
N=64, dt=2.5:  Completed in 40s ✅
N=128, dt=2.5: Allowed 240s (2x multiplier), completed in 160s ✅
Result: Proper convergence check
```

## Example: Early Termination

**Old Behavior:**
```
dt=40.0, N=256: Hit 120s limit after 125s ❌
dt=20.0, N=256: Still runs, hits 120s limit after 250s ❌
Result: Wasted 250 seconds
```

**New Behavior:**
```
dt=40.0, N=256: Hit 120s limit after 125s ❌
dt=20.0, N=256: Skipped (first failed) ⏭️
Result: Saved 250 seconds of computation
```

## Usage in Dummy Solutions

The dummy solution scripts now benefit automatically:

```python
def find_convergent_N(profile, N, dt, tolerance_rmse, ...):
    for i in range(max_iteration_num):
        # Run current resolution
        cost_i, _, completed = get_results(profile, N=current_N, dt=dt)

        if not completed:
            # Hit wall time - this N is too large
            # No point trying larger N
            break

        if i > 0:
            # Compare with previous
            params1 = {"N": previous_N, "dt": dt}
            params2 = {"N": current_N, "dt": dt}

            # This automatically:
            # 1. Skips if first failed
            # 2. Allows 2x time for second
            is_converged, _, _, rmse = compare_solutions(
                profile, params1, params2, tolerance_rmse
            )
```

## Configuration

Wall time limits remain in config files as defaults:
```yaml
# run_configs/hasegawa_mima_linear/p1.yaml
max_wall_time: 60  # seconds

# run_configs/hasegawa_mima_nonlinear/p1.yaml
max_wall_time: 120  # seconds
```

These can be overridden per-run as needed.

## Testing

To test the improvements:

```bash
# Clear cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Run with default limit
python runners/hasegawa_mima_nonlinear.py --config-name=p1 N=256 dt=2.5

# Run with extended limit
python runners/hasegawa_mima_nonlinear.py --config-name=p1 N=256 dt=2.5 max_wall_time=300

# Run with no limit
python runners/hasegawa_mima_nonlinear.py --config-name=p1 N=256 dt=2.5 max_wall_time=null
```
