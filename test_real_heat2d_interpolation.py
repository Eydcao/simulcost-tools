#!/usr/bin/env python3
"""
Test script using real heat 2D simulation results with the enhanced comparison function.
Uses existing get_res_heat_steady_2d API and compare_res_heat_steady_2d with interpolation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wrappers.heat_steady_2d import get_res_heat_steady_2d, compare_res_heat_steady_2d

def main():
    print("Testing real heat 2D simulation results with enhanced comparison function...")
    
    # Use different dx values to get different grid resolutions
    profile = "p1"
    relax = 0.6
    error_threshold = 1e-8
    t_init = 0.5
    
    # Fine grid (smaller dx = higher resolution)
    dx_fine = 0.005
    
    # Coarse grid (larger dx = lower resolution) 
    dx_coarse = 0.01

    print(f"Loading fine grid simulation (dx={dx_fine})...")
    T_fine, x_fine, y_fine, iter_fine, metadata_fine = get_res_heat_steady_2d(
        profile, dx_fine, relax, error_threshold, t_init
    )
    
    print(f"Loading coarse grid simulation (dx={dx_coarse})...")
    T_coarse, x_coarse, y_coarse, iter_coarse, metadata_coarse = get_res_heat_steady_2d(
        profile, dx_coarse, relax, error_threshold, t_init
    )
    
    print(f"Fine grid: {len(x_fine)}x{len(y_fine)} = {len(x_fine)*len(y_fine)} points")
    print(f"Coarse grid: {len(x_coarse)}x{len(y_coarse)} = {len(x_coarse)*len(y_coarse)} points")
    print(f"Fine grid iterations: {iter_fine}")
    print(f"Coarse grid iterations: {iter_coarse}")
    print(f"Fine grid cost: {metadata_fine['cost']}")
    print(f"Coarse grid cost: {metadata_coarse['cost']}")
    
    # Test the enhanced comparison function
    print("\nTesting enhanced comparison function...")
    rmse_tolerance = 0.1  # Relaxed tolerance for demonstration
    converged, metrics1, metrics2, rmse = compare_res_heat_steady_2d(
        profile, dx_fine, relax, error_threshold, t_init,    # Fine grid params
        profile, dx_coarse, relax, error_threshold, t_init,  # Coarse grid params
        rmse_tolerance
    )
    
    print(f"Comparison result: converged={converged}, RMSE={rmse:.6f}")
    
    # Now manually apply the same interpolation logic to visualize
    from scipy.interpolate import RegularGridInterpolator
    
    # Determine which grid is coarser (fewer points)
    grid_fine_size = len(x_fine) * len(y_fine)
    grid_coarse_size = len(x_coarse) * len(y_coarse)
    
    if grid_fine_size > grid_coarse_size:
        # Fine grid is finer, downsample to coarse grid's resolution
        print("Downsampling fine grid to coarse resolution for visualization...")
        interpolator = RegularGridInterpolator((x_fine, y_fine), T_fine, method='linear', bounds_error=False, fill_value=None)
        X_coarse_mesh, Y_coarse_mesh = np.meshgrid(x_coarse, y_coarse, indexing='ij')
        T_fine_interp = interpolator(np.stack([X_coarse_mesh.ravel(), Y_coarse_mesh.ravel()], axis=1)).reshape(X_coarse_mesh.shape)
        
        # Results for visualization
        T1_final = T_fine_interp
        T2_final = T_coarse
        x_common, y_common = x_coarse, y_coarse
        
        # Remove boundary conditions for internal comparison
        if len(x_common) > 2 and len(y_common) > 2:
            T1_internal = T1_final[1:-1, 1:-1]
            T2_internal = T2_final[1:-1, 1:-1]
            x_internal = x_common[1:-1]
            y_internal = y_common[1:-1]
        else:
            T1_internal = T1_final
            T2_internal = T2_final
            x_internal = x_common
            y_internal = y_common
            
        # Calculate relative difference like in comparison function
        eps = 1e-12
        def denom(a, b):
            std_a = np.std(a)
            std_b = np.std(b)
            return 0.5 * (np.abs(std_a) + np.abs(std_b)) + eps
        
        T_diff = np.abs(T1_internal - T2_internal) / denom(T1_internal, T2_internal)
        rmse_manual = np.sqrt(np.mean(T_diff**2))
        
        print(f"Manual RMSE calculation: {rmse_manual:.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Real Heat 2D Simulation Comparison\nProfile: {profile}, relax: {relax}, error_threshold: {error_threshold}, t_init: {t_init}', fontsize=12)
    
    # Plot 1: Original fine grid
    X_fine_mesh, Y_fine_mesh = np.meshgrid(x_fine, y_fine, indexing='ij')
    im1 = axes[0,0].contourf(X_fine_mesh, Y_fine_mesh, T_fine, levels=20, cmap='hot')
    axes[0,0].set_title(f'Fine Grid (dx={dx_fine})\n{len(x_fine)}x{len(y_fine)} points')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Plot 2: Original coarse grid  
    X_coarse_mesh, Y_coarse_mesh = np.meshgrid(x_coarse, y_coarse, indexing='ij')
    im2 = axes[0,1].contourf(X_coarse_mesh, Y_coarse_mesh, T_coarse, levels=20, cmap='hot')
    axes[0,1].set_title(f'Coarse Grid (dx={dx_coarse})\n{len(x_coarse)}x{len(y_coarse)} points')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Plot 3: Interpolated fine grid on coarse resolution
    im3 = axes[0,2].contourf(X_coarse_mesh, Y_coarse_mesh, T_fine_interp, levels=20, cmap='hot')
    axes[0,2].set_title('Fine→Coarse Interpolated')
    axes[0,2].set_xlabel('x')
    axes[0,2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Plot 4: Absolute difference between coarse and interpolated
    diff_abs = np.abs(T_coarse - T_fine_interp)
    im4 = axes[1,0].contourf(X_coarse_mesh, Y_coarse_mesh, diff_abs, levels=20, cmap='viridis')
    axes[1,0].set_title(f'Absolute Difference\nMax: {np.max(diff_abs):.4f}')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1,0])
    
    # Plot 5: Internal area comparison (boundary removed)
    if len(x_internal) > 0 and len(y_internal) > 0:
        X_internal, Y_internal = np.meshgrid(x_internal, y_internal, indexing='ij')
        im5 = axes[1,1].contourf(X_internal, Y_internal, T1_internal, levels=20, cmap='hot')
        axes[1,1].set_title('Internal Area (BC Removed)')
        axes[1,1].set_xlabel('x')
        axes[1,1].set_ylabel('y')
        plt.colorbar(im5, ax=axes[1,1])
        
        # Plot 6: Relative difference in internal area
        im6 = axes[1,2].contourf(X_internal, Y_internal, T_diff, levels=20, cmap='plasma')
        axes[1,2].set_title(f'Relative Diff\nRMSE={rmse:.4f}')
        axes[1,2].set_xlabel('x')
        axes[1,2].set_ylabel('y')
        plt.colorbar(im6, ax=axes[1,2])
    else:
        axes[1,1].text(0.5, 0.5, 'No internal\narea available', ha='center', va='center')
        axes[1,2].text(0.5, 0.5, 'No internal\narea available', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('real_heat2d_interpolation_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualization saved to real_heat2d_interpolation_test.png")
    print("\nReal simulation comparison demonstrates:")
    print("1. Actual heat 2D solutions at different resolutions (dx values)")
    print("2. Grid downsampling interpolation between different dx simulations")
    print("3. Enhanced comparison function with boundary condition removal")
    print("4. Relative RMSE calculation as implemented in the wrapper")
    print(f"5. Convergence result: {converged} (RMSE {rmse:.4f} vs tolerance {rmse_tolerance})")

if __name__ == "__main__":
    main()