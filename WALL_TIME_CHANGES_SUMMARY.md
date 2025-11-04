# Wall Time Implementation Summary

## Changes Made

### 1. Reduced record_dt for Nonlinear Solver
- Changed `record_dt` from 10000 to 1000 in all nonlinear configs (p1-p5)
- This reduces simulation time by 10x for nonlinear runs
- Expected max wall time reduced from ~765s to ~77s

### 2. Added Wall Time Tracking to Base Solver
**File:** `solvers/base_solver.py`
- Added `wall_time_start`, `wall_time_total`, `wall_time_exceeded` attributes
- Added `max_wall_time` parameter support from config
- Wall time tracking starts at `run()` and records total time
- Checks wall time limit during simulation loop

### 3. Implemented Hard Wall Time Constraints
**Config Changes:**
- Linear solvers (`run_configs/hasegawa_mima_linear/*.yaml`): `max_wall_time: 60` seconds
- Nonlinear solvers (`run_configs/hasegawa_mima_nonlinear/*.yaml`): `max_wall_time: 120` seconds

**Behavior:**
- Simulation stops if wall time exceeds limit
- Sets `wall_time_exceeded = True` flag
- Still runs `post_process()` to save partial results

### 4. Updated Solver Meta.json Output
**Files:**
- `solvers/hasegawa_mima_linear.py`
- `solvers/hasegawa_mima_nonlinear.py`

**New fields in meta.json:**
```json
{
  "wall_time_total": 12.34,
  "wall_time_exceeded": false,
  "simulation_completed": true
}
```

### 5. Updated Wrappers to Detect Incomplete Simulations

**Linear Wrapper** (`wrappers/hasegawa_mima_linear.py`):
- `get_error_metric()` checks `simulation_completed` flag
- Returns `None` if simulation didn't complete

**Nonlinear Wrapper** (`wrappers/hasegawa_mima_nonlinear.py`):
- `get_results()` now returns `(cost, results_list, simulation_completed)`
- `compare_solutions()` checks completion status
- `format_param_for_path()` fixed to use `.2e` format consistently

### 6. Updated Dummy Solutions
**File:** `dummy_sols/hasegawa_mima_nonlinear.py`
- Updated to handle 3-value return from `get_results()`

## How to Use

### Running Simulations
Simulations automatically track wall time and respect limits set in configs:

```bash
python runners/hasegawa_mima_linear.py --config-name=p1 N=256 dt=5.0
python runners/hasegawa_mima_nonlinear.py --config-name=p1 N=256 dt=2.5
```

### Checking Completion Status
```python
import json

with open('sim_res/path/to/meta.json') as f:
    meta = json.load(f)

if meta['simulation_completed']:
    print(f"Completed in {meta['wall_time_total']:.2f}s")
else:
    print(f"Hit wall time limit at {meta['wall_time_total']:.2f}s")
```

### Adjusting Wall Time Limits
Edit the config files:
```yaml
max_wall_time: 120  # seconds
```

## Expected Performance

### Linear Solver
- Typical runs: 8-15 seconds
- Max wall time limit: 60 seconds (4x safety margin)
- All current tasks should complete within limit

### Nonlinear Solver
- After reducing record_dt to 1/10:
  - Previous max: 765s → New max: ~77s
  - Typical runs: 20-80 seconds
  - Max wall time limit: 120 seconds (1.5-2x safety margin)

## Testing
Tested with sample simulations:
- Linear p1: Completed in 7.5s, all fields present in meta.json
- Nonlinear p1: Completed in 1.9s, all fields present in meta.json

## Next Steps
To rerun checkouts with new settings:
```bash
# Clear Python cache first
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Run checkouts
python checkouts/hasegawa_mima_nonlinear.py
python checkouts/hasegawa_mima_linear.py
```
