# Euler 1D Checkout Document

## Summary

- **Solver**: 1D Euler equations for compressible inviscid flow
- **Method**: 2nd order MUSCL scheme with Roe flux and generalized superbee limiter
- **Strategy**: Benchmark approach with 5 profiles
- **Benchmark**: Sod shock tube problem; TODO add more once done
- **Target Parameters**: 4 (CFL, n_space, k, beta)
- **Precision Levels**: 3 (high defined, medium/low TODO fill)

## Task Distribution

Current configuration generates:

- **CFL + n_space** (iterative): 5 profiles × 9 non-target combos × 2 target params = 90 tasks
- **k + beta** (0-shot): 5 profiles × 3 non-target combos × 2 target params = 30 tasks
- **Total per precision**: 120 tasks
- **Total tasks**: 360 tasks

## Configuration file

Refer to the **Configuration file**: `euler_1d_config.yaml`

## Dummy Solution Generation

Refer to the script: `checkouts/euler_1d_dummy_generation.py`
