#!/usr/bin/env python3
"""Extract NRMSE values from comparison run output."""
import re
import sys
sys.path.insert(0, '.')

from wrappers.euler_2d import get_res_euler_2d, interpolate_field_2d_interior, _extract_interior_cells, _read_grid_dims_from_meta
from solvers.utils import compute_nrmse

def extract_nrmse_for_comparison(profile, testcase, nx_coarse, nx_fine, cfl, cg_tolerance, frames):
    """Extract NRMSE values for a specific resolution comparison."""
    
    # Get grid dimensions
    _, ny_coarse = _read_grid_dims_from_meta(profile, nx_coarse, cfl, cg_tolerance)
    _, ny_fine = _read_grid_dims_from_meta(profile, nx_fine, cfl, cg_tolerance)
    aspect_ratio = ny_coarse / nx_coarse
    
    # Load results
    results_coarse = get_res_euler_2d(profile, testcase, nx_coarse, 0, 20, cfl, cg_tolerance)
    results_fine = get_res_euler_2d(profile, testcase, nx_fine, 0, 20, cfl, cg_tolerance)
    
    nrmse_data = []
    
    for frame in frames:
        if frame not in results_coarse or frame not in results_fine:
            continue
            
        data_coarse = results_coarse[frame]
        data_fine = results_fine[frame]
        
        # Extract interior cells
        density_coarse = _extract_interior_cells(data_coarse['density'], nx_coarse, ny_coarse)
        pressure_coarse = _extract_interior_cells(data_coarse['pressure'], nx_coarse, ny_coarse)
        
        density_fine = _extract_interior_cells(data_fine['density'], nx_fine, ny_fine)
        pressure_fine = _extract_interior_cells(data_fine['pressure'], nx_fine, ny_fine)
        
        # Interpolate
        density_interp = interpolate_field_2d_interior(
            density_coarse, nx_coarse, ny_coarse, nx_fine, ny_fine, aspect_ratio
        )
        pressure_interp = interpolate_field_2d_interior(
            pressure_coarse, nx_coarse, ny_coarse, nx_fine, ny_fine, aspect_ratio
        )
        
        # Compute NRMSE
        density_nrmse = compute_nrmse(density_interp, density_fine)
        pressure_nrmse = compute_nrmse(pressure_interp, pressure_fine)
        avg_nrmse = (density_nrmse + pressure_nrmse) / 2.0
        
        nrmse_data.append({
            'frame': frame,
            'density': density_nrmse,
            'pressure': pressure_nrmse,
            'average': avg_nrmse
        })
    
    return nrmse_data

# Configuration
profile = "p3"
testcase = 2
cfl = 0.5
cg_tolerance = 1e-7
frames = [5, 10, 15, 20]

comparisons = [
    (64, 128, "64→128"),
    (128, 256, "128→256"),
    (256, 512, "256→512"),
]

print("="*80)
print("NRMSE Summary for P3 Adjacent Resolution Comparisons")
print("="*80)

for nx_coarse, nx_fine, label in comparisons:
    print(f"\n{label}:")
    print("-"*80)
    
    nrmse_data = extract_nrmse_for_comparison(
        profile, testcase, nx_coarse, nx_fine, cfl, cg_tolerance, frames
    )
    
    print(f"{'Frame':<8} {'Density':<12} {'Pressure':<12} {'Average':<12}")
    print("-"*80)
    for data in nrmse_data:
        print(f"{data['frame']:<8} {data['density']:<12.6e} {data['pressure']:<12.6e} {data['average']:<12.6e}")

print("\n" + "="*80)
