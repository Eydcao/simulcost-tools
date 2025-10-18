# Euler 2D Checkout Summary

## Overview
This document summarizes the Euler 2D solver checkout process with the updated interpolation logic and corrected ny calculation.

## Key Fixes Implemented

### 1. Critical Bug Fix: C++ Integer Division Behavior
**Problem**: Python wrapper was using floor division (`//`) which behaves differently than C++ integer division for negative numbers.

**Root Cause**:
```cpp
// C++ code (gas_2d.cpp)
int Ny = std::round(aspect_ratio_y * Nx);
EulerSim::Array<int, dim, 1> bbmin{ -Nx / 2, -Ny / 2 };  // C++ truncates toward zero
EulerSim::Array<int, dim, 1> bbmax{ Nx / 2, Ny / 2 };
// Actual grid size = bbmax - bbmin
```

- C++ integer division: `-11 / 2 = -5` (truncates toward zero)
- Python floor division: `-11 // 2 = -6` (rounds toward negative infinity)

**Solution**: Implemented `_compute_ny_from_cpp_logic()` function:
```python
def _compute_ny_from_cpp_logic(aspect_ratio, nx):
    Ny_temp = int(round(aspect_ratio * nx))
    bbmin_y = int(-Ny_temp / 2)  # Use int(x/2) not x//2
    bbmax_y = int(Ny_temp / 2)
    ny_actual = bbmax_y - bbmin_y
    return ny_actual
```

**Impact**: For p2 (aspect_ratio=0.333):
- nx=32: Correct ny=10 (was incorrectly computing ny=11)
- nx=16: Correct ny=4 (was incorrectly computing ny=5)

### 2. Ghost Cell Handling
- VTK exports (nx+4) × (ny+4) points with ghost_layer=2
- Interior cells: [2:nx+2, 2:ny+2] in 2D grid
- Same grid comparison: Extract interior cells from both, exclude ghosts
- Different grid comparison: Interpolate coarser → finer, then extract interior from finer grid

### 3. 2D Interpolation with Proper Coordinates
- Interpolates from source VTK grid to target interior grid
- Uses scipy's RegularGridInterpolator with bilinear interpolation
- Coordinates match C++ cell-centered positions: `(i + 0.5) * dx`
- Accounts for VTK origin at `(-1.5 + bbmin) * dx`

## Precision Level Configuration

Updated tolerance levels to create difficulty gradient:

| Level  | Tolerance RMSE | Description                          |
|--------|----------------|--------------------------------------|
| Low    | 0.12           | Relaxed - captures most convergences |
| Medium | 0.08           | Moderate - requires finer grids      |
| High   | 0.06           | Stringent - very fine grids needed   |

## P1 (central_boom_2d) Checkout Results

**Profile Details**:
- Domain: 1.0 × 1.0 (square)
- Test case: 0 (central explosion)
- Frames: 0-20, record_dt=0.075

**Results**:
```
Total tasks: 36
Successful: 36 (100%)
Failed: 0 (0%)
Total cost: 47,588,586,240
Total time: 19.42s

By Precision Level:
  low:    12/12 (100%)
  medium: 12/12 (100%)
  high:   12/12 (100%)
```

**Analysis**:
- All tasks converged successfully
- 100% success rate across all precision levels
- Higher precision tasks required more iterations/finer grids
- Pattern observed:
  - Low precision: Often converges at 16→32 or 32→64
  - Medium precision: Often requires 32→64 or 64→128
  - High precision: Requires 64→128 or higher

**RMSE Values Observed**:
- n_grid_x refinement: 0.09-0.16 (16→32), 0.05-0.11 (32→64)
- cfl reduction: 0.058-0.063 (0.5→0.25)
- cg_tolerance: 0.00004-0.00007 (very tight already)

## P2 and P3 Checkout Status

**Currently Running** (as of session end):
- P2 (stair_flow_2d): aspect_ratio=0.333, testcase=1
- P3 (Mach_Diamond): aspect_ratio=0.5, testcase=2

Progress: Running simulation for p2 with nx=512 (high-resolution grid)

Results will be available in:
- `dataset/euler_2d/successful/tasks.json`
- `dataset/euler_2d/failed/tasks.json`
- `dataset/euler_2d/summary.json`

## Test Validation

All tests passed with updated logic:

**test_euler_2d_interpolation.py**:
✅ Grid: nx=32, ny=10 (correctly computed)
✅ VTK points: 36×14=504 (matches expected)
✅ Interpolation: 32×10 → 16×4 successful

**test_euler_2d_comparison.py**:
✅ Test 1: Same grid (32×10 vs 32×10) - RMSE=0
✅ Test 2: Different grids (32×10 vs 16×4) - Interpolation works
✅ Test 3: Self-comparison (p1 32×32) - RMSE=0

## Files Modified

1. `wrappers/euler_2d.py`:
   - Added `_compute_ny_from_cpp_logic()` function
   - Updated `run_sim_euler_2d()`, `get_res_euler_2d()`, `compare_res_euler_2d()`
   - Fixed interior cell extraction for different grid comparisons
   - Updated interpolation to return interior cells only

2. `checkouts/euler_2d.yaml`:
   - Updated precision levels: 0.12, 0.08, 0.06

3. `test_euler_2d_interpolation.py`:
   - Uses `_compute_ny_from_cpp_logic()` for correct ny calculation

4. `test_euler_2d_comparison.py`:
   - Comprehensive test for ghost cell handling and interpolation

## Next Steps

1. **Monitor P2 and P3 checkout completion**
   - Check final success rates by precision level
   - Verify reasonable trend (if any failures occur)

2. **Review Cost vs Precision Trend**
   - Even with 100% success, check if higher precision requires more iterations
   - Analyze optimal parameter values found at each precision level

3. **Validate Dataset Quality**
   - Ensure tasks represent realistic optimization scenarios
   - Confirm diversity in parameter combinations

## Commands to Check Progress

```bash
# Check p2/p3 checkout progress
tail -f checkout_p2_p3.log

# View final summary
cat dataset/euler_2d/summary.json | python -m json.tool

# Check success/failure breakdown
wc -l dataset/euler_2d/successful/tasks.json dataset/euler_2d/failed/tasks.json
```

## Session Status

✅ P1 checkout completed successfully (100% success rate)
🏃 P2 and P3 checkout running in background
📊 Updated interpolation logic validated and working correctly
🐛 Critical C++ integer division bug fixed

---
*Generated during Claude Code session on 2025-10-17*
*User went to bed - checkout process continues in background*
