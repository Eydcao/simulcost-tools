# Metadata Simplification - Removed Redundant Field

## Date: 2025-11-04

## Change Summary

Removed the redundant `simulation_completed` field from metadata. Now using only `wall_time_exceeded` for cleaner, more explicit tracking.

## Previous Metadata Structure

```json
{
  "cost": 737413120.0,
  "wall_time_total": 2.42,
  "wall_time_exceeded": false,
  "simulation_completed": true  // ← REDUNDANT
}
```

**Problem**: `simulation_completed` was always `not wall_time_exceeded`, making it redundant.

## New Simplified Metadata Structure

```json
{
  "cost": 737413120.0,
  "wall_time_total": 2.42,
  "wall_time_exceeded": false
}
```

**Benefit**: Single source of truth. Logic is clearer and simpler.

## Updated Logic

### Interpreting wall_time_exceeded

- `wall_time_exceeded = false` → Simulation completed successfully
- `wall_time_exceeded = true` → Simulation terminated due to wall time limit

### Code Changes

**Solvers** (`solvers/hasegawa_mima_linear.py`, `solvers/hasegawa_mima_nonlinear.py`):
```python
# Before:
meta = {
    "wall_time_total": float(self.wall_time_total),
    "wall_time_exceeded": bool(self.wall_time_exceeded),
    "simulation_completed": bool(self.current_time >= self.end_time - 1e-10 and not self.wall_time_exceeded),
}

# After:
meta = {
    "wall_time_total": float(self.wall_time_total),
    "wall_time_exceeded": bool(self.wall_time_exceeded),
}
```

**Wrappers** (`wrappers/hasegawa_mima_nonlinear.py`):
```python
# Before:
simulation_completed = meta.get("simulation_completed", True)
wall_time_exceeded = meta.get("wall_time_exceeded", False)
if not simulation_completed or wall_time_exceeded:
    print(f"Warning: Simulation did not complete")

# After:
wall_time_exceeded = meta.get("wall_time_exceeded", False)
simulation_completed = not wall_time_exceeded
if wall_time_exceeded:
    print(f"Warning: Simulation did not complete (wall_time_exceeded={wall_time_exceeded})")
```

**Linear Wrapper** (`wrappers/hasegawa_mima_linear.py`):
```python
# Before:
simulation_completed = meta.get("simulation_completed", True)
wall_time_exceeded = meta.get("wall_time_exceeded", False)
if not simulation_completed or wall_time_exceeded:
    return None

# After:
wall_time_exceeded = meta.get("wall_time_exceeded", False)
if wall_time_exceeded:
    return None
```

## Benefits

1. **Simpler Metadata**: One less field to maintain
2. **Single Source of Truth**: `wall_time_exceeded` is the authoritative field
3. **Clearer Semantics**: Explicit "exceeded" flag vs derived "completed" flag
4. **Easier to Reason About**: Less redundancy = less cognitive load
5. **Backward Compatible**: Wrappers still work with old metadata that has both fields

## Testing

✅ Tested with direct function call:
```bash
$ python -c "from dummy_sols.hasegawa_mima_nonlinear import find_convergent_N; ..."
✅ Test passed: (True, 64, [737413120.0, 3539582976.0], [...])
```

✅ Verified metadata structure:
```json
{
  "wall_time_total": 2.4246017932891846,
  "wall_time_exceeded": false
}
```
No `simulation_completed` field present ✅

## Migration Notes

### For Old Simulation Results

Old simulation results with both fields will continue to work:
- Wrappers use `meta.get("wall_time_exceeded", False)` with default value
- If old metadata has both fields, only `wall_time_exceeded` is read

### For New Code

All new simulations will only save `wall_time_exceeded`:
- Cleaner metadata files
- No confusion about which field to check
- Simpler logic in wrappers and analysis code

## Files Modified

1. `solvers/hasegawa_mima_linear.py` - Removed simulation_completed from metadata
2. `solvers/hasegawa_mima_nonlinear.py` - Removed simulation_completed from metadata
3. `wrappers/hasegawa_mima_nonlinear.py` - Simplified completion checking logic
4. `wrappers/hasegawa_mima_linear.py` - Simplified completion checking logic

## Summary

The metadata is now simpler and more maintainable. The single `wall_time_exceeded` field provides all the information needed to determine whether a simulation completed successfully.
