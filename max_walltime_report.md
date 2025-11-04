# Maximum Wall Time Measurements

Wall time measurements for the most computationally expensive cases.

## Hasegawa-Mima Linear

| Profile | Case Type | N | dt | cg_atol | Wall Time (s) | Status |
|---------|-----------|---|----|---------|--------------:|--------|
| p1 | Max N | 256 | 40.0 | 1.0e+00 | 11.85 | ✅ |
| p1 | Min dt | 128 | 5.0 | 1.0e+00 | 11.25 | ✅ |
| p2 | Max N | 256 | 40.0 | 1.0e+00 | 11.58 | ✅ |
| p2 | Min dt | 32 | 5.0 | 1.0e+00 | 8.26 | ✅ |
| p3 | Max N | 256 | 40.0 | 1.0e-01 | 14.94 | ✅ |
| p3 | Min dt | 64 | 5.0 | 1.0e+00 | 9.80 | ✅ |
| p4 | Max N | 256 | 40.0 | 1.0e-01 | 14.80 | ✅ |
| p4 | Min dt | 128 | 5.0 | 1.0e-01 | 12.49 | ✅ |

## Hasegawa-Mima Nonlinear

| Profile | Case Type | N | dt | Wall Time (s) | Status |
|---------|-----------|---|----|--------------:|--------|
| p1 | Max N | 128 | 20.0 | 23.04 | ✅ |
| p1 | Min dt | 64 | 2.5 | 41.53 | ✅ |
| p2 | Max N | 256 | 2.5 | 759.72 | ✅ |
| p2 | Min dt | 64 | 2.5 | 41.14 | ✅ |
| p3 | Max N | 128 | 2.5 | 161.74 | ✅ |
| p3 | Min dt | 64 | 2.5 | 41.67 | ✅ |
| p4 | Max N | 256 | 2.5 | 764.21 | ✅ |
| p4 | Min dt | 64 | 2.5 | 42.92 | ✅ |
| p5 | Max N | 256 | 2.5 | 765.87 | ✅ |
| p5 | Min dt | 64 | 2.5 | 41.58 | ✅ |

## Summary

**Linear solver:**
- Runs: 8
- Max: 14.94s, Min: 8.26s, Avg: 11.87s

**Nonlinear solver:**
- Runs: 10
- Max: 765.87s, Min: 23.04s, Avg: 268.34s

*Generated: 2025-11-03 20:53:45*
