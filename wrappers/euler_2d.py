import os
import subprocess
import numpy as np
import json
from scipy.interpolate import RegularGridInterpolator
import meshio


def _compute_nrmse_maxabs(field_low, field_high, eps=1e-12):
    # Compute RMSE
    diff = field_low - field_high
    rmse = np.sqrt(np.mean(diff**2))

    # Normalize by max absolute value of high-res field
    max_abs_high = np.max(np.abs(field_high)) + eps  # Add eps to avoid division by zero

    return rmse / max_abs_high


def _find_runner_path():
    """Automatically find the correct path to euler_2d.py runner."""
    cwd = os.getcwd()

    possible_paths = []

    # If working from project root
    if cwd.endswith("SimulCost-Bench") or cwd.endswith("costsci-tools"):
        possible_paths.extend(["runners/euler_2d.py", "costsci_tools/runners/euler_2d.py"])
    # If working from costsci_tools/ subdirectory
    elif "costsci_tools" in cwd:
        possible_paths.extend(["runners/euler_2d.py", "../runners/euler_2d.py", "costsci_tools/runners/euler_2d.py"])

    # Add generic fallback paths
    possible_paths.extend(
        [
            "runners/euler_2d.py",
            "costsci_tools/runners/euler_2d.py",
            "./runners/euler_2d.py",
            "../runners/euler_2d.py",
            "../../runners/euler_2d.py",
        ]
    )

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


def _read_vtk_structured(vtk_path, nx, ny):
    """
    Read binary VTK structured grid file and extract INTERIOR-ONLY data.

    This function reads the raw data, which includes ghost cells,
    and immediately extracts the interior data.

    IMPORTANT: Pressure is stored on vertices (at i*dx positions),
    while other fields are stored on cell centers (at (i+0.5)*dx positions).

    Args:
        vtk_path (str): Path to the VTK file.
        nx (int): Number of *interior* cells in x.
        ny (int): Number of *interior* cells in y.

    Returns:
        dict: Dictionary containing INTERIOR-ONLY field data.
              - Cell-centered fields (density, velocity): shape (ny, nx)
              - Vertex-centered fields (pressure): shape (ny+1, nx+1)
    """
    mesh = meshio.read(vtk_path)
    data = {}

    ghost_layer = 2
    nx_pts = nx + 2 * ghost_layer
    ny_pts = ny + 2 * ghost_layer

    if mesh.point_data:
        for field_name, field_values_raw in mesh.point_data.items():

            # Determine if this is a vertex-centered field (pressure)
            is_vertex = field_name.lower() == "pressure" or field_name.lower() == "p"

            # --- Helper to extract interior 2D field ---
            def extract_cell_centered(raw_flat_field):
                # Reshape to 2D (y-major, x-fastest)
                field_2d = raw_flat_field.reshape((ny_pts, nx_pts))
                # Extract interior cells [ghost_layer : ghost_layer+ny, ghost_layer : ghost_layer+nx]
                interior_2d = field_2d[ghost_layer : ghost_layer + ny, ghost_layer : ghost_layer + nx]
                return interior_2d

            def extract_vertex_centered(raw_flat_field):
                # Reshape to 2D (y-major, x-fastest)
                field_2d = raw_flat_field.reshape((ny_pts, nx_pts))
                # Extract interior vertices [ghost_layer : ghost_layer+ny+1, ghost_layer : ghost_layer+nx+1]
                # One extra vertex in each direction to bound the cells
                interior_2d = field_2d[ghost_layer : ghost_layer + ny + 1, ghost_layer : ghost_layer + nx + 1]
                return interior_2d

            # ------------------------------------------------

            # Handle different shapes
            if field_values_raw.ndim == 2:
                if field_values_raw.shape[1] == 1:
                    # Scalar field stored as (N, 1) - flatten to 1D
                    if is_vertex:
                        data[field_name] = extract_vertex_centered(field_values_raw.ravel())
                    else:
                        data[field_name] = extract_cell_centered(field_values_raw.ravel())
                elif field_values_raw.shape[1] == 3:
                    # Vector field (3 components) - always cell-centered
                    data[f"{field_name}_x"] = extract_cell_centered(field_values_raw[:, 0])
                    data[f"{field_name}_y"] = extract_cell_centered(field_values_raw[:, 1])
                    # z-component available as field_values[:, 2] if needed
            elif field_values_raw.ndim == 1:
                # Already 1D scalar field
                if is_vertex:
                    data[field_name] = extract_vertex_centered(field_values_raw)
                else:
                    data[field_name] = extract_cell_centered(field_values_raw)

    return data


def get_res_euler_2d(profile, n_grid_x, cfl, cg_tolerance):
    """
    Load all time frames for a given parameter set, triggering a simulation
    if results are missing.

    Returns a tuple: (results, cost)
    - results (dict): Dict of {frame: data_dict}. Data is INTERIOR-ONLY
                      and stored in 2D (ny, nx) arrays.
    - cost (float): Simulation cost.
    """
    # --- 1. Define paths ---
    cfl_str = f"{cfl:.3f}" if cfl is not None else "default"
    cgtol_str = f"{cg_tolerance:.1e}" if cg_tolerance is not None else "default"
    dir_path = f"sim_res/euler_2d/{profile}_cfl_{cfl_str}_cgtol_{cgtol_str}_nx_{n_grid_x}/"
    meta_path = os.path.join(dir_path, "meta.json")
    vtk_dir = os.path.join(dir_path, "vtk")

    # --- 2. Check for valid cache or run simulation ---
    if not os.path.exists(meta_path):
        print(f"No meta.json. Running new simulation: n_grid_x={n_grid_x}, cfl={cfl}, cg_tolerance={cg_tolerance}")
        runner_path = _find_runner_path()
        cmd = f"python {runner_path} --config-name={profile} n_grid_x={n_grid_x} cfl={cfl} cg_tolerance={cg_tolerance}"
        subprocess.run(cmd, shell=True, check=True)
    else:
        # Check if meta.json is valid (has 'cost')
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" not in meta:
                print(
                    f"Invalid meta.json. Running new simulation: n_grid_x={n_grid_x}, cfl={cfl}, cg_tolerance={cg_tolerance}"
                )
                runner_path = _find_runner_path()
                cmd = f"python {runner_path} --config-name={profile} n_grid_x={n_grid_x} cfl={cfl} cg_tolerance={cg_tolerance}"
                subprocess.run(cmd, shell=True, check=True)
            else:
                print(f"Using existing simulation results from {dir_path}")

    # --- 3. Load meta and VTK results (files *must* exist now) ---
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]
        nx = meta["parameters"]["n_grid_x"]
        ny = meta["parameters"]["n_grid_y"]

    results = {}
    files = [f for f in os.listdir(vtk_dir) if f.startswith("gas_frame_") and f.endswith(".vtk")]
    files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

    for file_name in files:
        file_path = os.path.join(vtk_dir, file_name)
        frame_number = int(file_name.split("_")[2].split(".")[0])

        # Pass nx, ny to reader, which returns INTERIOR-ONLY 2D data
        data = _read_vtk_structured(file_path, nx, ny)
        results[frame_number] = data

    # --- MODIFIED: Return only results and cost ---
    return results, cost


def _interpolate_field_2d_interior(field_src_2d, nx_src, ny_src, nx_tgt, ny_tgt, is_vertex=False):
    """
    Interpolates a 2D field from source grid to target grid.

    Handles both cell-centered and vertex-centered fields correctly.

    Args:
        field_src_2d: 2D array of INTERIOR data only
                      - Cell-centered: shape (ny_src, nx_src), at (i+0.5)*dx
                      - Vertex-centered: shape (ny_src+1, nx_src+1), at i*dx
        nx_src: Number of interior cells in x-direction for source
        ny_src: Number of interior cells in y-direction for source
        nx_tgt: Number of interior cells in x-direction for target
        ny_tgt: Number of interior cells in y-direction for target
        is_vertex: True if field is vertex-centered (like pressure)

    Returns:
        Interpolated 2D field on target grid
        - Cell-centered: shape (ny_tgt, nx_tgt)
        - Vertex-centered: shape (ny_tgt+1, nx_tgt+1)
    """
    # Grid spacing
    dx_src = 1.0 / nx_src
    dy_src = dx_src

    if is_vertex:
        # Vertex-centered: defined at vertices i*dx
        x_src = np.array([i * dx_src for i in range(nx_src + 1)])
        y_src = np.array([j * dy_src for j in range(ny_src + 1)])
    else:
        # Cell-centered: defined at cell centers (i+0.5)*dx
        x_src = np.array([(i + 0.5) * dx_src for i in range(nx_src)])
        y_src = np.array([(j + 0.5) * dy_src for j in range(ny_src)])

    # Create interpolator (RegularGridInterpolator expects (y, x) order for 2D)
    interp = RegularGridInterpolator(
        (y_src, x_src),
        field_src_2d,
        method="linear",
        bounds_error=False,
        fill_value=None,  # Use extrapolation at boundaries
    )

    # Create coordinates for target grid
    dx_tgt = 1.0 / nx_tgt
    dy_tgt = dx_tgt

    if is_vertex:
        # Vertex-centered target points
        x_tgt = np.array([i * dx_tgt for i in range(nx_tgt + 1)])
        y_tgt = np.array([j * dy_tgt for j in range(ny_tgt + 1)])
    else:
        # Cell-centered target points
        x_tgt = np.array([(i + 0.5) * dx_tgt for i in range(nx_tgt)])
        y_tgt = np.array([(j + 0.5) * dy_tgt for j in range(ny_tgt)])

    # Create meshgrid for target points (indexing='ij' gives (ny, nx) shape)
    yy_tgt, xx_tgt = np.meshgrid(y_tgt, x_tgt, indexing="ij")

    # Interpolate
    points = np.stack([yy_tgt.ravel(), xx_tgt.ravel()], axis=-1)

    if is_vertex:
        field_tgt_2d = interp(points).reshape((ny_tgt + 1, nx_tgt + 1))
    else:
        field_tgt_2d = interp(points).reshape((ny_tgt, nx_tgt))

    return field_tgt_2d


def compare_res_euler_2d(
    profile1, n_grid_x1_in, profile2, n_grid_x2_in, rmse_tolerance, cfl1, cg_tolerance1, cfl2, cg_tolerance2
):
    """Compare two sets of results using relative error norms and physical metrics.

    Returns:
        converged (bool): True if RMSE tolerance is met.
        metrics1 (dict): Metrics for case 1.
        metrics2 (dict): Metrics for case 2.
        rmse (float): RMSE of relative difference.
    """
    # --- MODIFIED: get_res now returns only (results, cost) ---
    res1, _ = get_res_euler_2d(profile1, n_grid_x1_in, cfl1, cg_tolerance1)
    res2, _ = get_res_euler_2d(profile2, n_grid_x2_in, cfl2, cg_tolerance2)

    # --- MODIFIED: Strict check for matching frames ---
    frames1 = set(res1.keys())
    frames2 = set(res2.keys())
    if frames1 != frames2:
        raise ValueError(
            f"Frame sets do not match.\n"
            f"Frames 1 ({len(frames1)}): {sorted(list(frames1))}\n"
            f"Frames 2 ({len(frames2)}): {sorted(list(frames2))}"
        )

    if len(frames1) == 0:
        raise ValueError("No frames found in simulation results")

    # Compare density and pressure fields at final frame
    common_frames = sorted(list(frames1))  # Use sorted list from set
    final_frame = common_frames[-1]
    data1 = res1[final_frame]
    data2 = res2[final_frame]

    # --- MODIFIED: Data is already 2D interior-only ---
    density1_interior = data1["density"]
    density2_interior = data2["density"]
    pressure1_interior = data1["pressure"]
    pressure2_interior = data2["pressure"]

    # --- MODIFIED: Infer nx, ny from shape ---
    # Shape is (ny, nx) due to (y_src, x_src) interpolation order
    ny1, nx1 = density1_interior.shape
    ny2, nx2 = density2_interior.shape

    # --- MODIFIED: Compare shapes ---
    if (ny1, nx1) == (ny2, nx2):
        # Same grid size - direct comparison of interior cells
        print(f"Same grid size detected ({nx1}x{ny1}) - comparing directly")
        # Use NRMSE: both fields are same resolution, use either for normalization
        density_nrmse = _compute_nrmse_maxabs(density1_interior, density2_interior)
        pressure_nrmse = _compute_nrmse_maxabs(pressure1_interior, pressure2_interior)
    else:
        # Different grid sizes - interpolate coarser to finer, then compare
        # Now using the clean _interpolate_field_2d_interior function
        if (nx1 * ny1) < (nx2 * ny2):
            # data1 is coarser, interpolate to data2's finer grid
            print(f"Interpolating coarse grid ({nx1}x{ny1}) to fine grid ({nx2}x{ny2})")

            # Interpolate cell-centered density
            density1_interp = _interpolate_field_2d_interior(density1_interior, nx1, ny1, nx2, ny2, is_vertex=False)
            # Interpolate vertex-centered pressure
            pressure1_interp = _interpolate_field_2d_interior(pressure1_interior, nx1, ny1, nx2, ny2, is_vertex=True)

            # Use NRMSE: data2 is finer (higher res), use it for normalization
            density_nrmse = _compute_nrmse_maxabs(density1_interp, density2_interior)
            pressure_nrmse = _compute_nrmse_maxabs(pressure1_interp, pressure2_interior)
        else:
            # data2 is coarser, interpolate to data1's finer grid
            print(f"Interpolating coarse grid ({nx2}x{ny2}) to fine grid ({nx1}x{ny1})")

            # Interpolate cell-centered density
            density2_interp = _interpolate_field_2d_interior(density2_interior, nx2, ny2, nx1, ny1, is_vertex=False)
            # Interpolate vertex-centered pressure
            pressure2_interp = _interpolate_field_2d_interior(pressure2_interior, nx2, ny2, nx1, ny1, is_vertex=True)

            # Use NRMSE: data1 is finer (higher res), use it for normalization
            density_nrmse = _compute_nrmse_maxabs(density2_interp, density1_interior)
            pressure_nrmse = _compute_nrmse_maxabs(pressure2_interp, pressure1_interior)

    # Average NRMSE across channels (density and pressure)
    rmse = (density_nrmse + pressure_nrmse) / 2.0

    # Convergence criteria
    converged = rmse < rmse_tolerance

    print(f"RMSE (relative): {rmse}")

    return converged, rmse


if __name__ == "__main__":
    ps = ["p1", "p2", "p3", "p4", "p5"]
    for p in ps:
        print(compare_res_euler_2d(p, 32, p, 64, 0.01, 0.25, 1e-6, 0.25, 1e-6))
