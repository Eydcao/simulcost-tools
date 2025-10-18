#!/usr/bin/env python
"""Test script to verify 2D mesh reading and interpolation for euler_2d."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wrappers.euler_2d import run_sim_euler_2d, get_res_euler_2d, interpolate_field_2d, _compute_ny_from_cpp_logic, _get_aspect_ratio_from_config
import numpy as np

# Test with p2 (aspect_ratio=0.333, non-1) to verify row/col major ordering
profile = "p2"
testcase = 1  # stair_flow_2d
n_grid_x = 32
aspect_ratio = _get_aspect_ratio_from_config(profile)
ny = _compute_ny_from_cpp_logic(aspect_ratio, n_grid_x)

print("="*60)
print("Testing Euler 2D mesh reading and interpolation")
print("="*60)
print(f"Profile: {profile}")
print(f"Testcase: {testcase}")
print(f"Grid: nx={n_grid_x}, ny={ny}")
print(f"Aspect ratio: {aspect_ratio}")
print()

# Run a small simulation
print("Step 1: Running simulation...")
cfl = 0.5
cg_tolerance = 1e-7
cost = run_sim_euler_2d(profile, testcase, n_grid_x, start_frame=0, end_frame=5, cfl=cfl, cg_tolerance=cg_tolerance)
print(f"Cost: {cost}")
print()

# Load results
print("Step 2: Loading results...")
results = get_res_euler_2d(profile, testcase, n_grid_x, start_frame=0, end_frame=5, cfl=cfl, cg_tolerance=cg_tolerance)
print(f"Loaded {len(results)} frames")
print()

# Check first frame data
if len(results) > 0:
    frame_0 = results[0]
    print("Step 3: Analyzing frame 0 data structure...")
    print(f"Fields available: {list(frame_0.keys())}")
    print()

    if 'density' in frame_0:
        density = frame_0['density']
        print(f"Density array shape: {density.shape}")
        print(f"Density array length: {len(density)}")

        # VTK STRUCTURED_POINTS: (nx+4) × (ny+4) points with ghost_layer=2
        expected_nx_pts = n_grid_x + 4
        expected_ny_pts = ny + 4
        expected_total = expected_nx_pts * expected_ny_pts

        print(f"Expected VTK grid: {expected_nx_pts} x {expected_ny_pts} = {expected_total} points")
        print(f"Match: {len(density) == expected_total}")
        print()

        # Reshape and check
        if len(density) == expected_total:
            # Try reshape with different orderings
            print("Step 4: Testing array ordering...")
            print(f"Attempting reshape to ({expected_ny_pts}, {expected_nx_pts}) [y-major, x-fastest]...")
            try:
                density_2d = density.reshape((expected_ny_pts, expected_nx_pts))
                print(f"  Success! Shape: {density_2d.shape}")
                print(f"  Min: {np.min(density_2d):.6f}, Max: {np.max(density_2d):.6f}, Mean: {np.mean(density_2d):.6f}")
            except Exception as e:
                print(f"  Failed: {e}")
            print()

            # Test interpolation to a coarser grid
            print("Step 5: Testing interpolation...")
            n_grid_x_coarse = 16
            ny_coarse = _compute_ny_from_cpp_logic(aspect_ratio, n_grid_x_coarse)
            print(f"Interpolating from {n_grid_x}x{ny} to {n_grid_x_coarse}x{ny_coarse}...")

            try:
                density_interp = interpolate_field_2d(
                    density, n_grid_x, ny, n_grid_x_coarse, ny_coarse, aspect_ratio
                )
                print(f"  Success! Interpolated array length: {len(density_interp)}")
                print(f"  Expected length: {n_grid_x_coarse * ny_coarse}")
                print(f"  Min: {np.min(density_interp):.6f}, Max: {np.max(density_interp):.6f}, Mean: {np.mean(density_interp):.6f}")
                print(f"  Match: {len(density_interp) == n_grid_x_coarse * ny_coarse}")
            except Exception as e:
                print(f"  Failed: {e}")
                import traceback
                traceback.print_exc()
            print()
        else:
            print(f"ERROR: Size mismatch! Got {len(density)}, expected {expected_total}")
            print("Cannot proceed with interpolation test.")
    else:
        print("ERROR: 'density' field not found in results")
else:
    print("ERROR: No frames loaded")

print("="*60)
print("Test complete")
print("="*60)
