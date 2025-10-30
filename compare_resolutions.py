#!/usr/bin/env python3
"""
Unified script to compare Euler 2D simulations at different resolutions.

This script:
1. Extracts interior cells (removes ghost layers)
2. Interpolates coarse grid to fine grid using bilinear interpolation
3. Computes NRMSE (Normalized Root Mean Square Error)
4. Exports VTK files for visualization
5. Generates PNG comparison plots (2x2 subfigures)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wrappers.euler_2d import (
    get_res_euler_2d,
    interpolate_field_2d_interior,
    _extract_interior_cells,
    _read_grid_dims_from_meta
)
from solvers.utils import compute_nrmse


def export_vtk_structured(filename, field_2d, nx, ny, aspect_ratio, field_name="scalar"):
    """
    Export a 2D field as VTK STRUCTURED_POINTS for visualization.

    Args:
        filename: Output VTK filename
        field_2d: 2D numpy array (ny, nx) of cell-centered values
        nx, ny: Grid dimensions
        aspect_ratio: Domain aspect ratio
        field_name: Name of the scalar field
    """
    dx = 1.0 / nx
    dy = aspect_ratio / ny

    with open(filename, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"2D {field_name} field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write(f"ORIGIN 0.0 0.0 0.0\n")
        f.write(f"SPACING {dx} {dy} 1.0\n")
        f.write(f"POINT_DATA {nx * ny}\n")
        f.write(f"SCALARS {field_name} float\n")
        f.write("LOOKUP_TABLE default\n")

        # Write data (flatten in x-fastest order for VTK)
        for j in range(ny):
            for i in range(nx):
                f.write(f"{field_2d[j, i]}\n")

    print(f"  Exported {filename}")


def plot_2x2_comparison(output_path, field_name,
                        density_2d, pressure_2d, vx_2d, vy_2d,
                        nx, ny, aspect_ratio, title_suffix=""):
    """
    Create a 2x2 subplot showing density, pressure, vx, vy.

    Args:
        output_path: Path to save PNG file
        field_name: Base name for the fields
        density_2d, pressure_2d, vx_2d, vy_2d: 2D arrays (ny, nx)
        nx, ny: Grid dimensions
        aspect_ratio: Domain aspect ratio
        title_suffix: Additional text for the title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    extent = [0, 1, 0, aspect_ratio]

    # Density
    im0 = axes[0, 0].imshow(density_2d, origin='lower', extent=extent,
                            aspect='auto', cmap='viridis')
    axes[0, 0].set_title(f'Density ({nx}x{ny})')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])

    # Pressure
    im1 = axes[0, 1].imshow(pressure_2d, origin='lower', extent=extent,
                            aspect='auto', cmap='plasma')
    axes[0, 1].set_title(f'Pressure ({nx}x{ny})')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])

    # Velocity X
    im2 = axes[1, 0].imshow(vx_2d, origin='lower', extent=extent,
                            aspect='auto', cmap='RdBu_r')
    axes[1, 0].set_title(f'Velocity X ({nx}x{ny})')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0])

    # Velocity Y
    im3 = axes[1, 1].imshow(vy_2d, origin='lower', extent=extent,
                            aspect='auto', cmap='RdBu_r')
    axes[1, 1].set_title(f'Velocity Y ({nx}x{ny})')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1])

    plt.suptitle(f'{field_name}{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


def compare_resolutions(profile, testcase, nx_coarse, nx_fine, cfl, cg_tolerance,
                       output_dir, test_frames):
    """
    Compare two resolutions: extract interior, interpolate, compute NRMSE, export VTK and PNG.

    Args:
        profile: Profile name (e.g., "p3")
        testcase: Test case number
        nx_coarse, nx_fine: Grid resolutions to compare
        cfl: CFL number
        cg_tolerance: CG solver tolerance
        output_dir: Directory to save outputs
        test_frames: List of frame numbers to process
    """
    print(f"\n{'='*80}")
    print(f"Comparing {nx_coarse} vs {nx_fine}")
    print(f"{'='*80}")

    # Get grid dimensions from meta.json
    _, ny_coarse = _read_grid_dims_from_meta(profile, nx_coarse, cfl, cg_tolerance)
    _, ny_fine = _read_grid_dims_from_meta(profile, nx_fine, cfl, cg_tolerance)

    aspect_ratio = ny_coarse / nx_coarse

    print(f"Configuration:")
    print(f"  Profile: {profile}, Testcase: {testcase}")
    print(f"  Aspect ratio: {aspect_ratio:.6f}")
    print(f"  Coarse grid: nx={nx_coarse}, ny={ny_coarse}")
    print(f"  Fine grid: nx={nx_fine}, ny={ny_fine}")
    print(f"  CFL: {cfl}, CG tolerance: {cg_tolerance}")

    # Load results
    print(f"\nLoading simulation results...")
    results_coarse = get_res_euler_2d(profile, testcase, nx_coarse, 0, 20, cfl, cg_tolerance)
    results_fine = get_res_euler_2d(profile, testcase, nx_fine, 0, 20, cfl, cg_tolerance)

    print(f"  Coarse grid: {len(results_coarse)} frames loaded")
    print(f"  Fine grid: {len(results_fine)} frames loaded")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")

    # Process each frame
    for frame in test_frames:
        if frame not in results_coarse or frame not in results_fine:
            print(f"\nSkipping frame {frame} (not available in both simulations)")
            continue

        print(f"\n{'-'*80}")
        print(f"Processing Frame {frame}")
        print(f"{'-'*80}")

        data_coarse = results_coarse[frame]
        data_fine = results_fine[frame]

        # Extract interior cells
        density_coarse = _extract_interior_cells(data_coarse['density'], nx_coarse, ny_coarse)
        pressure_coarse = _extract_interior_cells(data_coarse['pressure'], nx_coarse, ny_coarse)
        vx_coarse = _extract_interior_cells(data_coarse['velocity_x'], nx_coarse, ny_coarse)
        vy_coarse = _extract_interior_cells(data_coarse['velocity_y'], nx_coarse, ny_coarse)

        density_fine = _extract_interior_cells(data_fine['density'], nx_fine, ny_fine)
        pressure_fine = _extract_interior_cells(data_fine['pressure'], nx_fine, ny_fine)
        vx_fine = _extract_interior_cells(data_fine['velocity_x'], nx_fine, ny_fine)
        vy_fine = _extract_interior_cells(data_fine['velocity_y'], nx_fine, ny_fine)

        # Interpolate coarse to fine grid
        print(f"Interpolating coarse ({nx_coarse}x{ny_coarse}) to fine ({nx_fine}x{ny_fine})...")
        density_interp = interpolate_field_2d_interior(
            density_coarse, nx_coarse, ny_coarse, nx_fine, ny_fine, aspect_ratio
        )
        pressure_interp = interpolate_field_2d_interior(
            pressure_coarse, nx_coarse, ny_coarse, nx_fine, ny_fine, aspect_ratio
        )
        vx_interp = interpolate_field_2d_interior(
            vx_coarse, nx_coarse, ny_coarse, nx_fine, ny_fine, aspect_ratio
        )
        vy_interp = interpolate_field_2d_interior(
            vy_coarse, nx_coarse, ny_coarse, nx_fine, ny_fine, aspect_ratio
        )

        # Compute NRMSE (using fine grid for normalization)
        density_nrmse = compute_nrmse(density_interp, density_fine)
        pressure_nrmse = compute_nrmse(pressure_interp, pressure_fine)
        vx_nrmse = compute_nrmse(vx_interp, vx_fine)
        vy_nrmse = compute_nrmse(vy_interp, vy_fine)

        # Average NRMSE across all fields
        avg_nrmse = (density_nrmse + pressure_nrmse + vx_nrmse + vy_nrmse) / 4.0

        print(f"\nNRMSE Results:")
        print(f"  Density:  {density_nrmse:.6e}")
        print(f"  Pressure: {pressure_nrmse:.6e}")
        print(f"  Vx:       {vx_nrmse:.6e}")
        print(f"  Vy:       {vy_nrmse:.6e}")
        print(f"  Average:  {avg_nrmse:.6e}")

        # Reshape to 2D for export
        density_coarse_2d = density_coarse.reshape((ny_coarse, nx_coarse))
        pressure_coarse_2d = pressure_coarse.reshape((ny_coarse, nx_coarse))
        vx_coarse_2d = vx_coarse.reshape((ny_coarse, nx_coarse))
        vy_coarse_2d = vy_coarse.reshape((ny_coarse, nx_coarse))

        density_fine_2d = density_fine.reshape((ny_fine, nx_fine))
        pressure_fine_2d = pressure_fine.reshape((ny_fine, nx_fine))
        vx_fine_2d = vx_fine.reshape((ny_fine, nx_fine))
        vy_fine_2d = vy_fine.reshape((ny_fine, nx_fine))

        density_interp_2d = density_interp.reshape((ny_fine, nx_fine))
        pressure_interp_2d = pressure_interp.reshape((ny_fine, nx_fine))
        vx_interp_2d = vx_interp.reshape((ny_fine, nx_fine))
        vy_interp_2d = vy_interp.reshape((ny_fine, nx_fine))

        # Export VTK files
        print(f"\nExporting VTK files...")
        export_vtk_structured(
            f"{output_dir}/density_coarse_frame_{frame:03d}.vtk",
            density_coarse_2d, nx_coarse, ny_coarse, aspect_ratio, "density"
        )
        export_vtk_structured(
            f"{output_dir}/density_fine_frame_{frame:03d}.vtk",
            density_fine_2d, nx_fine, ny_fine, aspect_ratio, "density"
        )
        export_vtk_structured(
            f"{output_dir}/density_interp_frame_{frame:03d}.vtk",
            density_interp_2d, nx_fine, ny_fine, aspect_ratio, "density_interpolated"
        )

        # Export PNG plots (2x2 subfigures)
        print(f"\nExporting PNG plots...")
        plot_2x2_comparison(
            f"{output_dir}/coarse_frame_{frame:03d}.png",
            "Coarse Grid", density_coarse_2d, pressure_coarse_2d, vx_coarse_2d, vy_coarse_2d,
            nx_coarse, ny_coarse, aspect_ratio, f" - Frame {frame}"
        )
        plot_2x2_comparison(
            f"{output_dir}/fine_frame_{frame:03d}.png",
            "Fine Grid (Actual)", density_fine_2d, pressure_fine_2d, vx_fine_2d, vy_fine_2d,
            nx_fine, ny_fine, aspect_ratio, f" - Frame {frame}"
        )
        plot_2x2_comparison(
            f"{output_dir}/interp_frame_{frame:03d}.png",
            "Fine Grid (Interpolated)", density_interp_2d, pressure_interp_2d, vx_interp_2d, vy_interp_2d,
            nx_fine, ny_fine, aspect_ratio, f" - Frame {frame}, NRMSE={avg_nrmse:.3e}"
        )


def main():
    """Main function to compare all p3 resolutions."""

    print("="*80)
    print("Euler 2D Resolution Comparison Tool")
    print("="*80)

    # Configuration
    profile = "p3"
    testcase = 2
    cfl = 0.5
    cg_tolerance = 1e-7
    test_frames = [5, 10, 15, 20]

    # Define resolution pairs to compare
    resolution_pairs = [
        (64, 128),
        (128, 256),
        (256, 512),
        (64, 256),
        (64, 512),
        (128, 512),
    ]

    for nx_coarse, nx_fine in resolution_pairs:
        output_dir = f"comparison_{profile}_{nx_coarse}_{nx_fine}"
        try:
            compare_resolutions(
                profile, testcase, nx_coarse, nx_fine, cfl, cg_tolerance,
                output_dir, test_frames
            )
        except FileNotFoundError as e:
            print(f"\nSkipping {nx_coarse}→{nx_fine}: {e}")
        except Exception as e:
            print(f"\nError processing {nx_coarse}→{nx_fine}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("Resolution Comparison Complete!")
    print(f"{'='*80}")
    print("\nOutput directories created:")
    for nx_coarse, nx_fine in resolution_pairs:
        print(f"  - comparison_{profile}_{nx_coarse}_{nx_fine}/")
    print("\nFiles generated per comparison:")
    print("  - *_coarse_*.vtk: Coarse grid VTK files")
    print("  - *_fine_*.vtk: Fine grid VTK files")
    print("  - *_interp_*.vtk: Interpolated grid VTK files")
    print("  - coarse_*.png: 2x2 plots of coarse grid")
    print("  - fine_*.png: 2x2 plots of fine grid")
    print("  - interp_*.png: 2x2 plots of interpolated grid with NRMSE")


if __name__ == "__main__":
    main()
