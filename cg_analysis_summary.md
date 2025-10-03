# CG Residual Analysis: Resolution Sensitivity Study

## Executive Summary

This analysis reveals **dramatic differences** in CG solver behavior across grid resolutions (N=64, 128, 256) for the Hasegawa-Mima linear solver. The findings have critical implications for parameter optimization strategies.

## Key Findings

### 1. **CG Iteration Scaling**

- **N=64**: 3.6-7.0 avg iterations depending on cg_atol
- **N=128**: 6.5-13.5 avg iterations (2x increase)
- **N=256**: 19.9-26.4 avg iterations (2-3x further increase)

**Scaling follows theory**: CG iterations ~ O(√κ) ~ O(N) for Helmholtz operator

### 2. **Convergence Rate Degradation**

| Resolution | Convergence Rate (orders/iteration) | Pattern |
|------------|-------------------------------------|---------|
| N=64       | 0.96-1.01                          | Rapid, consistent |
| N=128      | 0.56-0.75                          | Moderate slowdown |
| N=256      | 0.18-0.19                          | **Severe degradation** |

**Critical insight**: N=256 requires ~5x more iterations per order of magnitude reduction!

### 3. **Cost Scaling Analysis**

| Transition | Theoretical (N²) | Observed | Extra Factor |
|------------|------------------|----------|--------------|
| 64→128     | 4.0x             | 6.9x     | 1.7x         |
| 128→256    | 4.0x             | 7.5x     | 1.9x         |

**Cost scaling**: O(N³⁺) due to both N² scaling per iteration AND increased iteration count

### 4. **Tolerance Sensitivity by Resolution**

#### N=64 (High Sensitivity)

- `cg_atol=1e-6` vs `1e-8`: **Identical** performance (over-solving)
- `cg_atol=1e-4`: **20% cost reduction**, final residual 3.2e-5
- `cg_atol=1e-3`: **44% cost reduction**, final residual 2.9e-4

#### N=128 (Moderate Sensitivity)

- `cg_atol=1e-6` vs `1e-8`: **Identical** performance
- `cg_atol=1e-4`: **21% cost reduction**, final residual 7.3e-5
- `cg_atol=1e-3`: **47% cost reduction**, final residual 6.7e-4

#### N=256 (Low Sensitivity)

- `cg_atol=1e-6` vs `1e-8`: **Identical** performance
- `cg_atol=1e-4`: **<1% cost reduction**, final residual 5.9e-5
- `cg_atol=1e-3`: **22% cost reduction**, final residual 5.3e-4

### 5. **Residual Trajectory Patterns**

#### N=64: Well-behaved convergence

```
3.83e-02 → 2.84e-03 → 2.89e-04 → 3.64e-05 → 5.37e-06
Reductions: 13.5x → 9.8x → 7.9x → 6.8x (consistent)
```

#### N=128: Moderate irregularity

```
7.73e-02 → 5.83e-03 → 6.10e-04 → 1.70e-04 → 8.19e-05 → ...
Reductions: 13.3x → 9.6x → 3.6x → 2.1x (slowing)
```

#### N=256: Highly irregular convergence

```
1.55e-01 → 1.17e-02 → 2.60e-03 → 1.40e-03 → 3.76e-03 → ...
Reductions: 13.2x → 4.5x → 1.9x → 0.4x (stalling!)
```

**Note**: N=256 shows **stagnation phases** where residual barely decreases or even increases!

## Optimization Recommendations

### 1. **Resolution-Dependent cg_atol Strategy**

```python
def get_optimal_cg_atol(N):
    if N <= 64:
        return 1e-4    # 20% savings, good accuracy
    elif N <= 128:
        return 1e-4    # 21% savings, good accuracy
    else:  # N >= 256
        return 1e-3    # 22% savings (only option for significant savings)
```

### 2. **Parameter Search Range Updates**

**Current range**: `[1e-8, 1e-3]` with 4-6 logarithmic points
**Recommended ranges**:

- **N ≤ 128**: `[1e-5, 1e-2]` with focus on `[1e-4, 1e-3]`
- **N ≥ 256**: `[1e-4, 1e-2]` with focus on `[1e-3, 1e-2]`

### 3. **Cost Model Updates**

Current cost model underestimates high-resolution scaling:

```python
# Old model
cost = cg_iterations_total * N**2

# Improved model accounting for condition number scaling
cost = cg_iterations_total * N**2 * condition_number_factor
where condition_number_factor ≈ (N/64)**0.5
```

### 4. **Convergence Criteria Adjustments**

For large systems (N≥256), consider:

- **Stagnation detection**: Stop if residual reduction < 10% for 3 consecutive iterations
- **Adaptive tolerance**: Start with relaxed tolerance, tighten if convergence is too slow
- **Maximum iteration limits**: Prevent runaway solves for poorly conditioned systems

## Implementation Impact

### For `dummy_sols/hasegawa_mima_linear.py`

**✅ IMPLEMENTED: Updated `find_optimal_cg_atol()`**:
1. **CHANGED**: Now uses fixed N and dt parameters (0-shot search)
2. **REMOVED**: Inner N search loop (was causing nested optimization)
3. **IMPROVED**: Direct cost comparison across cg_atol values
4. **SIMPLIFIED**: Returns single optimal cg_atol instead of (cg_atol, N) tuple

**Function signature changes**:
```python
# OLD (nested search)
find_optimal_cg_atol(profile, N, dt, tolerance_rmse, search_range_min,
                    search_range_max, search_range_slice_num,
                    multiplication_factor, max_iteration_num)

# NEW (fixed parameters)
find_optimal_cg_atol(profile, N, dt, tolerance_rmse, search_range_min,
                    search_range_max, search_range_slice_num)
```

### For `checkouts/hasegawa_mima_linear.yaml`

**✅ IMPLEMENTED: Updated configuration**:
```yaml
target_parameters:
  cg_atol:
    description: "Conjugate Gradient solver absolute tolerance - controls linear solver accuracy"
    search_type: "0-shot"
    search_range: [1e-6, 1e-2]  # Updated range based on sensitivity analysis
    search_range_slice_num: 4    # 4 logarithmically spaced values
    non_target_parameters:
      N: [64, 128, 256]        # Test across different resolutions
      dt: [5.0, 10.0, 20.0]    # Multiple time step sizes
```

**Key changes**:
- **Removed**: `multiplication_factor` and `max_iteration_num` (no longer needed)
- **Updated**: Search range from `[1e-8, 1e-3]` to `[1e-6, 1e-2]`
- **Added**: Multiple N values to test resolution sensitivity

## Conclusion

**The CG solver behavior is fundamentally different across resolutions**. Low-resolution systems (N≤128) offer high sensitivity to cg_atol tuning with 20-47% cost savings available. High-resolution systems (N≥256) show poor tolerance sensitivity and require more aggressive relaxation (cg_atol≥1e-3) to achieve meaningful savings.

**This analysis suggests the current one-size-fits-all approach to cg_atol optimization is suboptimal and should be replaced with resolution-adaptive strategies.**
