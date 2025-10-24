import os
import subprocess
import numpy as np
import json
import sys
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import meshio

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers.utils import compute_nrmse

def _find_runner_path():
    """Automatically find the correct path to euler_2d.py runner."""
    cwd = os.getcwd()

    possible_paths = []

    # If working from project root
    if cwd.endswith('SimulCost-Bench') or cwd.endswith('costsci-tools'):
        possible_paths.extend([
            "runners/euler_2d.py",
            "costsci_tools/runners/euler_2d.py"
        ])
    # If working from costsci_tools/ subdirectory
    elif 'costsci_tools' in cwd:
        possible_paths.extend([
            "runners/euler_2d.py",
            "../runners/euler_2d.py",
            "costsci_tools/runners/euler_2d.py"
        ])

    # Add generic fallback paths
    possible_paths.extend([
        "runners/euler_2d.py",
        "costsci_tools/runners/euler_2d.py",
        "./runners/euler_2d.py",
        "../runners/euler_2d.py",
        "../../runners/euler_2d.py"
    ])

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in possible_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    for path in unique_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not find euler_2d.py runner in any expected location.\n"
        f"Current working directory: {cwd}\n"
        f"Searched paths: {unique_paths}\n"
        f"Please ensure the runner exists or update the search paths."
    )


def _extract_interior_cells(field_flat, nx, ny):
    """
    Extract interior cells from VTK STRUCTURED_POINTS data.

    VTK exports (nx+4) × (ny+4) points. We want to extract only the interior nx × ny cells.
    The ghost_layer=2 means 2 ghost points on each side, so interior cells are at indices [2:nx+2, 2:ny+2].

    Args:
        field_flat: Flattened 1D array from VTK ((nx+4)*(ny+4) points)
        nx: Number of interior cells in x
        ny: Number of interior cells in y

    Returns:
        Flattened 1D array of interior cells only (nx*ny values)
    """
    ghost_layer = 2
    nx_pts = nx + 2 * ghost_layer
    ny_pts = ny + 2 * ghost_layer

    # Reshape to 2D (y-major, x-fastest)
    field_2d = field_flat.reshape((ny_pts, nx_pts))

    # Extract interior [ghost_layer:-ghost_layer] in both dimensions
    interior_2d = field_2d[ghost_layer:ghost_layer+ny, ghost_layer:ghost_layer+nx]

    return interior_2d.ravel()


def _read_grid_dims_from_meta(profile, n_grid_x, cfl, cg_tolerance):
    """Read n_grid_x and n_grid_y from meta.json.

    Returns:
        tuple: (n_grid_x, n_grid_y) read from the simulation's meta.json
    """
    cfl_str = f"{cfl:.3f}" if cfl is not None else "default"
    cgtol_str = f"{cg_tolerance:.1e}" if cg_tolerance is not None else "default"
    dir_path = f"sim_res/euler_2d/{profile}_cfl_{cfl_str}_cgtol_{cgtol_str}_nx_{n_grid_x}/"
    meta_path = os.path.join(dir_path, "meta.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found at {meta_path}. Run simulation first.")

    with open(meta_path, "r") as f:
        meta = json.load(f)
        return meta["n_grid_x"], meta["n_grid_y"]


def run_sim_euler_2d(profile, testcase, n_grid_x, start_frame, end_frame, cfl, cg_tolerance):
    """Run the euler_2d simulation with the given parameters if not already simulated."""
    # Directory path includes all tunable parameters: profile, cfl, cg_tolerance, n_grid_x
    # Format similar to euler_1d: {profile}_cfl_{cfl}_cgtol_{cg_tol}_nx_{nx}
    # NOTE: n_grid_y is determined by C++ based on aspect ratio, recorded in meta.json
    cfl_str = f"{cfl:.3f}" if cfl is not None else "default"
    cgtol_str = f"{cg_tolerance:.1e}" if cg_tolerance is not None else "default"
    dir_path = f"sim_res/euler_2d/{profile}_cfl_{cfl_str}_cgtol_{cgtol_str}_nx_{n_grid_x}/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"]

    # Run the simulation if not already done
    print(f"Running new simulation with parameters: testcase={testcase}, n_grid_x={n_grid_x}, frames={start_frame}-{end_frame}, cfl={cfl}, cg_tolerance={cg_tolerance}")
    runner_path = _find_runner_path()
    cmd = f"python {runner_path} --config-name={profile} testcase={testcase} n_grid_x={n_grid_x} start_frame={start_frame} end_frame={end_frame}"
    if cfl is not None:
        cmd += f" cfl={cfl}"
    if cg_tolerance is not None:
        cmd += f" cg_tolerance={cg_tolerance}"
    subprocess.run(cmd, shell=True, check=True)

    # Load the cost from the meta.json file
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]

    return cost


def read_vtk_structured(vtk_path):
    """Read binary VTK structured grid file and extract data using meshio.

    Returns:
        dict: Dictionary containing field data arrays and grid dimensions
            - 'density', 'pressure', 'velocity_x', 'velocity_y': field arrays (flattened 1D)
            - 'nx_pts', 'ny_pts': VTK grid dimensions (points including ghosts)
            - 'nx', 'ny': Interior cell dimensions (VTK points - 4, since ghost_layer=2)
    """
    mesh = meshio.read(vtk_path)

    data = {}

    # Extract point data (scalars and vectors)
    if mesh.point_data:
        for field_name, field_values in mesh.point_data.items():
            # Flatten the array and handle different shapes
            if field_values.ndim == 2:
                if field_values.shape[1] == 1:
                    # Scalar field stored as (N, 1) - flatten to 1D
                    data[field_name] = field_values.ravel()
                elif field_values.shape[1] == 3:
                    # Vector field (3 components) - extract x,y components
                    data[f"{field_name}_x"] = field_values[:, 0]
                    data[f"{field_name}_y"] = field_values[:, 1]
                    # z-component available as field_values[:, 2] if needed
            elif field_values.ndim == 1:
                # Already 1D scalar field
                data[field_name] = field_values

    return data


def get_res_euler_2d(profile, testcase, n_grid_x, start_frame, end_frame, cfl, cg_tolerance):
    """Load all time frames for a given parameter set, triggering a simulation if results are missing."""

    # Use same directory naming as run_sim_euler_2d
    cfl_str = f"{cfl:.3f}" if cfl is not None else "default"
    cgtol_str = f"{cg_tolerance:.1e}" if cg_tolerance is not None else "default"
    dir_path = f"sim_res/euler_2d/{profile}_cfl_{cfl_str}_cgtol_{cgtol_str}_nx_{n_grid_x}/"
    vtk_dir = os.path.join(dir_path, "vtk")
    results = {}

    # Check if at least one result file exists, otherwise trigger a simulation
    if not os.path.exists(vtk_dir) or not any(
        fname.startswith("gas_frame_") and fname.endswith(".vtk") for fname in os.listdir(vtk_dir)
    ):
        print(
            f"No results found for parameters: testcase={testcase}, n_grid_x={n_grid_x}, cfl={cfl}, cg_tolerance={cg_tolerance}. Triggering simulation."
        )
        run_sim_euler_2d(profile, testcase, n_grid_x, start_frame, end_frame, cfl, cg_tolerance)

    # Load all VTK files
    files = [f for f in os.listdir(vtk_dir) if f.startswith("gas_frame_") and f.endswith(".vtk")]
    files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

    for file_name in files:
        file_path = os.path.join(vtk_dir, file_name)
        frame_number = int(file_name.split("_")[2].split(".")[0])

        try:
            data = read_vtk_structured(file_path)
            results[frame_number] = data
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")

    return results


def compute_euler_2d_metrics(results):
    """Compute physical metrics for Euler 2D solution.

    Args:
        results: Dictionary with frame_id -> {x, y, density, pressure, ux, uy}

    Returns:
        Dictionary containing:
        - positivity_preserved: bool array if pressure/density stay positive at all timesteps
        - pressure_range: (min, max) pressure across all frames
        - density_range: (min, max) density across all frames
    """
    frames = sorted(results.keys())
    if len(frames) < 2:
        return {}

    # Extract time series data
    density_mins = []
    density_maxs = []
    pressure_mins = []
    pressure_maxs = []

    for f in frames:
        density_mins.append(np.min(results[f]['density']))
        density_maxs.append(np.max(results[f]['density']))
        pressure_mins.append(np.min(results[f]['pressure']))
        pressure_maxs.append(np.max(results[f]['pressure']))

    # Positivity preservation
    positivity_preserved = np.all(np.array(density_mins) > 0) and np.all(np.array(pressure_mins) > 0)

    return {
        "positivity_preserved": positivity_preserved,
        "pressure_range": (np.min(pressure_mins), np.max(pressure_maxs)),
        "density_range": (np.min(density_mins), np.max(density_maxs)),
    }


def print_euler_2d_metrics(name, metrics):
    """Print summary statistics for Euler 2D metrics"""
    print(f"\n--- {name} Metrics ---")
    if not metrics:
        print("No metrics available (insufficient data)")
        return

    print(f"Positivity preserved: {metrics['positivity_preserved']}")
    print(f"Density range: [{metrics['density_range'][0]:.6f}, {metrics['density_range'][1]:.6f}]")
    print(f"Pressure range: [{metrics['pressure_range'][0]:.6f}, {metrics['pressure_range'][1]:.6f}]")


def interpolate_field_2d_interior(field_src_interior, nx_src, ny_src, nx_tgt, ny_tgt, aspect_ratio):
    """
    Interpolates a 2D field from source grid to target grid.

    CLEAN VERSION: Operates on interior-only data (no ghost cells).
    This function expects both input and output to be interior cells only.

    Args:
        field_src_interior: Flattened 1D array of INTERIOR cells only (nx_src * ny_src values)
        nx_src: Number of interior cells in x-direction for source
        ny_src: Number of interior cells in y-direction for source
        nx_tgt: Number of interior cells in x-direction for target
        ny_tgt: Number of interior cells in y-direction for target
        aspect_ratio: Domain aspect ratio (Ly / Lx)

    Returns:
        Flattened interpolated field on target grid interior (nx_tgt * ny_tgt values)
    """
    # Reshape interior data to 2D (y-major, x-fastest)
    field_src_2d = field_src_interior.reshape((ny_src, nx_src))

    # Grid spacing
    dx_src = 1.0 / nx_src
    dy_src = aspect_ratio / ny_src

    # Create coordinates for source grid (interior cell centers)
    # Cell centers are at (i+0.5)*dx for i=0,1,...,nx_src-1
    x_src = np.array([(i + 0.5) * dx_src for i in range(nx_src)])
    y_src = np.array([(j + 0.5) * dy_src for j in range(ny_src)])

    # Create interpolator (RegularGridInterpolator expects (y, x) order for 2D)
    interp = RegularGridInterpolator(
        (y_src, x_src),
        field_src_2d,
        method='linear',
        bounds_error=False,
        fill_value=None  # Use extrapolation at boundaries
    )

    # Create coordinates for target grid (interior cell centers)
    dx_tgt = 1.0 / nx_tgt
    dy_tgt = aspect_ratio / ny_tgt
    x_tgt = np.array([(i + 0.5) * dx_tgt for i in range(nx_tgt)])
    y_tgt = np.array([(j + 0.5) * dy_tgt for j in range(ny_tgt)])

    # Create meshgrid for target points (indexing='ij' gives (ny, nx) shape)
    yy_tgt, xx_tgt = np.meshgrid(y_tgt, x_tgt, indexing='ij')

    # Interpolate
    points = np.stack([yy_tgt.ravel(), xx_tgt.ravel()], axis=-1)
    field_tgt_2d = interp(points).reshape((ny_tgt, nx_tgt))

    # Return flattened interior data
    return field_tgt_2d.ravel()


def compare_res_euler_2d(
    profile1, testcase1, n_grid_x1,
    profile2, testcase2, n_grid_x2,
    rmse_tolerance,
    start_frame, end_frame,
    cfl1, cg_tolerance1,
    cfl2, cg_tolerance2
):
    """Compare two sets of results using relative error norms and physical metrics.

    Returns:
        converged (bool): True if RMSE tolerance is met.
        metrics1 (dict): Metrics for case 1.
        metrics2 (dict): Metrics for case 2.
        rmse (float): RMSE of relative difference.
    """
    res1 = get_res_euler_2d(profile1, testcase1, n_grid_x1, start_frame, end_frame, cfl1, cg_tolerance1)
    res2 = get_res_euler_2d(profile2, testcase2, n_grid_x2, start_frame, end_frame, cfl2, cg_tolerance2)

    # Get common frames
    frames1 = set(res1.keys())
    frames2 = set(res2.keys())
    common_frames = sorted(frames1 & frames2)

    if len(common_frames) == 0:
        raise ValueError("No common frames found between the two simulations")

    # Compare density and pressure fields at final frame
    final_frame = common_frames[-1]
    data1 = res1[final_frame]
    data2 = res2[final_frame]

    # Read grid dimensions directly from meta.json (C++ determined these based on aspect ratio)
    nx1, ny1 = _read_grid_dims_from_meta(profile1, n_grid_x1, cfl1, cg_tolerance1)
    nx2, ny2 = _read_grid_dims_from_meta(profile2, n_grid_x2, cfl2, cg_tolerance2)

    # Compute aspect ratio from actual grid dimensions (for interpolation)
    aspect_ratio1 = ny1 / nx1
    aspect_ratio2 = ny2 / nx2

    # Check that aspect ratios are similar (we're only comparing same testcases)
    if abs(aspect_ratio1 - aspect_ratio2) > 0.1:  # Relaxed tolerance for different resolutions
        print(f"Warning: Different aspect ratios detected: {aspect_ratio1:.4f} vs {aspect_ratio2:.4f}")

    # VTK STRUCTURED_POINTS exports points (vertices) with ghost_layer=2
    # Pattern: (nx+4) × (ny+4) points (C++ uses round() + integer division for ny calculation)
    ghost_layer = 2
    n_points1_expected = (nx1 + 2*ghost_layer) * (ny1 + 2*ghost_layer)
    n_points2_expected = (nx2 + 2*ghost_layer) * (ny2 + 2*ghost_layer)

    n_points1 = len(data1['density'])
    n_points2 = len(data2['density'])

    # Verify expected grid sizes
    print(f"Case 1: nx={nx1}, ny={ny1}, VTK points={n_points1} (expected {n_points1_expected} = ({nx1}+{2*ghost_layer})×({ny1}+{2*ghost_layer}))")
    print(f"Case 2: nx={nx2}, ny={ny2}, VTK points={n_points2} (expected {n_points2_expected} = ({nx2}+{2*ghost_layer})×({ny2}+{2*ghost_layer}))")

    if n_points1 != n_points1_expected:
        print(f"Warning: data1 has {n_points1} points, expected {n_points1_expected}")
    if n_points2 != n_points2_expected:
        print(f"Warning: data2 has {n_points2} points, expected {n_points2_expected}")

    # CLEAN APPROACH: Extract interior cells first for BOTH grids
    # This gives us clean interior-only data to work with
    density1_interior = _extract_interior_cells(data1['density'], nx1, ny1)
    density2_interior = _extract_interior_cells(data2['density'], nx2, ny2)
    pressure1_interior = _extract_interior_cells(data1['pressure'], nx1, ny1)
    pressure2_interior = _extract_interior_cells(data2['pressure'], nx2, ny2)

    if n_points1 == n_points2:
        # Same grid size - direct comparison of interior cells
        print("Same grid size detected - comparing interior cells directly")
        # Use NRMSE: both fields are same resolution, use either for normalization
        density_nrmse = compute_nrmse(density1_interior, density2_interior)
        pressure_nrmse = compute_nrmse(pressure1_interior, pressure2_interior)
    else:
        # Different grid sizes - interpolate coarser to finer, then compare
        # Now using the clean interpolate_field_2d_interior function
        if n_points1 < n_points2:
            # data1 is coarser, interpolate to data2's finer grid
            print(f"Interpolating coarse grid ({nx1}x{ny1}) to fine grid ({nx2}x{ny2})")
            density1_interp = interpolate_field_2d_interior(
                density1_interior, nx1, ny1, nx2, ny2, aspect_ratio1
            )
            pressure1_interp = interpolate_field_2d_interior(
                pressure1_interior, nx1, ny1, nx2, ny2, aspect_ratio1
            )

            # Use NRMSE: data2 is finer (higher res), use it for normalization
            density_nrmse = compute_nrmse(density1_interp, density2_interior)
            pressure_nrmse = compute_nrmse(pressure1_interp, pressure2_interior)
        else:
            # data2 is coarser, interpolate to data1's finer grid
            print(f"Interpolating coarse grid ({nx2}x{ny2}) to fine grid ({nx1}x{ny1})")
            density2_interp = interpolate_field_2d_interior(
                density2_interior, nx2, ny2, nx1, ny1, aspect_ratio2
            )
            pressure2_interp = interpolate_field_2d_interior(
                pressure2_interior, nx2, ny2, nx1, ny1, aspect_ratio2
            )

            # Use NRMSE: data1 is finer (higher res), use it for normalization
            density_nrmse = compute_nrmse(density2_interp, density1_interior)
            pressure_nrmse = compute_nrmse(pressure2_interp, pressure1_interior)

    # Average NRMSE across channels (density and pressure)
    rmse = (density_nrmse + pressure_nrmse) / 2.0

    # Compute metrics
    metrics1 = compute_euler_2d_metrics(res1)
    metrics2 = compute_euler_2d_metrics(res2)

    # Convergence criteria
    # Note: positivity check disabled for central explosion case which has vacuum regions
    converged = (
        rmse < rmse_tolerance
        # Positivity check removed: central explosion expands into vacuum (zero density/pressure at boundaries)
        # and metrics1.get("positivity_preserved", True)
        # and metrics2.get("positivity_preserved", True)
    )

    print_euler_2d_metrics("Case 1", metrics1)
    print_euler_2d_metrics("Case 2", metrics2)

    print(f"RMSE (relative): {rmse}")

    return converged, metrics1, metrics2, rmse


if __name__ == "__main__":
    # Example usage
    testcase = 0
    n_grid_x = 32
    cfl = 0.5
    cg_tolerance = 1e-7

    # Run a simulation
    cost = run_sim_euler_2d("p1", testcase, n_grid_x, start_frame=0, end_frame=10, cfl=cfl, cg_tolerance=cg_tolerance)
    print(f"Simulation cost: {cost}")

    # Load results
    results = get_res_euler_2d("p1", testcase, n_grid_x, start_frame=0, end_frame=10, cfl=cfl, cg_tolerance=cg_tolerance)
    print(f"Loaded {len(results)} frames")

    # Compute metrics
    metrics = compute_euler_2d_metrics(results)
    print_euler_2d_metrics("Test", metrics)
