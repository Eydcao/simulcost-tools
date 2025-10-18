#!/usr/bin/env python
"""Test script to verify ghost cell handling in comparison."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wrappers.euler_2d import run_sim_euler_2d, compare_res_euler_2d
import numpy as np

print("="*60)
print("Testing Euler 2D comparison with ghost cell handling")
print("="*60)
print()

# Test 1: Same grid size - should exclude ghost cells
print("Test 1: Same grid size (32x10) - should compare interior cells only")
print("-"*60)
profile1 = "p2"
profile2 = "p2"
testcase = 1
nx = 32
cfl = 0.5
cg_tolerance = 1e-7

# Ensure simulation exists
cost1 = run_sim_euler_2d(profile1, testcase, nx, start_frame=0, end_frame=5, cfl=cfl, cg_tolerance=cg_tolerance)
cost2 = run_sim_euler_2d(profile2, testcase, nx, start_frame=0, end_frame=5, cfl=cfl, cg_tolerance=cg_tolerance)

print(f"\nComparing {profile1} (nx={nx}) vs {profile2} (nx={nx})")
converged, metrics1, metrics2, rmse = compare_res_euler_2d(
    profile1, testcase, nx,
    profile2, testcase, nx,
    rmse_tolerance=0.05,
    start_frame=0, end_frame=5,
    cfl1=cfl, cg_tolerance1=cg_tolerance,
    cfl2=cfl, cg_tolerance2=cg_tolerance
)
print(f"Converged: {converged}, RMSE: {rmse:.6f}")
print()

# Test 2: Different grid sizes - should use interpolation
print("Test 2: Different grid sizes (32x10 vs 16x5) - should use interpolation")
print("-"*60)
nx1 = 32
nx2 = 16

# Ensure simulations exist
cost1 = run_sim_euler_2d(profile1, testcase, nx1, start_frame=0, end_frame=5, cfl=cfl, cg_tolerance=cg_tolerance)
cost2 = run_sim_euler_2d(profile2, testcase, nx2, start_frame=0, end_frame=5, cfl=cfl, cg_tolerance=cg_tolerance)

print(f"\nComparing {profile1} (nx={nx1}) vs {profile2} (nx={nx2})")
converged, metrics1, metrics2, rmse = compare_res_euler_2d(
    profile1, testcase, nx1,
    profile2, testcase, nx2,
    rmse_tolerance=0.05,
    start_frame=0, end_frame=5,
    cfl1=cfl, cg_tolerance1=cg_tolerance,
    cfl2=cfl, cg_tolerance2=cg_tolerance
)
print(f"Converged: {converged}, RMSE: {rmse:.6f}")
print()

# Test 3: Compare p1 (square) with itself - should be identical
print("Test 3: p1 (square domain) with itself - should be identical")
print("-"*60)
profile_square = "p1"
nx_square = 32

cost = run_sim_euler_2d(profile_square, testcase, nx_square, start_frame=0, end_frame=5, cfl=cfl, cg_tolerance=cg_tolerance)

print(f"\nComparing {profile_square} (nx={nx_square}) vs itself")
converged, metrics1, metrics2, rmse = compare_res_euler_2d(
    profile_square, testcase, nx_square,
    profile_square, testcase, nx_square,
    rmse_tolerance=0.001,
    start_frame=0, end_frame=5,
    cfl1=cfl, cg_tolerance1=cg_tolerance,
    cfl2=cfl, cg_tolerance2=cg_tolerance
)
print(f"Converged: {converged}, RMSE (should be ~0): {rmse:.10f}")
print()

print("="*60)
print("All tests complete")
print("="*60)
