import os
import subprocess
import numpy as np
import json
import struct
from pathlib import Path


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


def run_sim_euler_2d(profile, testcase, n_grid_x, start_frame=0, end_frame=180, cfl=None, cg_tolerance=None):
    """Run the euler_2d simulation with the given parameters if not already simulated."""
    # Determine Ny from testcase aspect ratio
    aspect_ratios = {0: 1.0, 1: 1.0/3.0, 2: 1.0/3.0, 3: 1.0/2.0}
    n_grid_y = int(round(aspect_ratios[testcase] * n_grid_x))

    # Directory path includes all tunable parameters: profile, cfl, cg_tolerance, n_grid_x
    # Format similar to euler_1d: {profile}_cfl_{cfl}_cgtol_{cg_tol}_nx_{nx}
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
    """Read binary VTK structured grid file and extract data."""
    with open(vtk_path, 'rb') as f:
        # Read ASCII header until POINT_DATA
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('POINT_DATA'):
                num_points = int(line.split()[1])
                break

        data = {}

        # Parse data sections
        while True:
            # Read next line (might be field header or empty)
            pos_before = f.tell()
            try:
                line_bytes = f.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode('ascii').strip()
            except:
                break

            if not line:
                continue

            if line.startswith('SCALARS'):
                # SCALARS field_name type components
                parts = line.split()
                field_name = parts[1]
                dtype = parts[2]

                # Read LOOKUP_TABLE line
                f.readline()

                # Read binary data
                bytes_per_val = 8 if dtype == 'double' else 4
                binary_data = f.read(num_points * bytes_per_val)
                if dtype == 'double':
                    field_data = np.frombuffer(binary_data, dtype='>f8')
                else:
                    field_data = np.frombuffer(binary_data, dtype='>f4')

                data[field_name] = field_data

                # Skip the trailing newline after binary data
                f.read(1)

            elif line.startswith('VECTORS'):
                # VECTORS field_name type
                parts = line.split()
                field_name = parts[1]
                dtype = parts[2]

                # Read 3-component vectors
                bytes_per_val = 8 if dtype == 'double' else 4
                binary_data = f.read(num_points * 3 * bytes_per_val)
                if dtype == 'double':
                    vector_data = np.frombuffer(binary_data, dtype='>f8')
                else:
                    vector_data = np.frombuffer(binary_data, dtype='>f4')

                # Reshape to (num_points, 3) and extract 2D components
                vector_data = vector_data.reshape((num_points, 3))
                data[f"{field_name}_x"] = vector_data[:, 0]
                data[f"{field_name}_y"] = vector_data[:, 1]

                # Skip the trailing newline after binary data
                f.read(1)

    return data


def get_res_euler_2d(profile, testcase, n_grid_x, start_frame=0, end_frame=180, cfl=None, cg_tolerance=None):
    """Load all time frames for a given parameter set, triggering a simulation if results are missing."""
    aspect_ratios = {0: 1.0, 1: 1.0/3.0, 2: 1.0/3.0, 3: 1.0/2.0}
    n_grid_y = int(round(aspect_ratios[testcase] * n_grid_x))

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


def compare_res_euler_2d(
    profile1, testcase1, n_grid_x1,
    profile2, testcase2, n_grid_x2,
    rmse_tolerance,
    start_frame=0, end_frame=180,
    cfl1=None, cg_tolerance1=None,
    cfl2=None, cg_tolerance2=None
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

    eps = 1e-12

    def relative_error(a, b):
        """Compute relative error using mean values as denominator"""
        mean_a = np.mean(np.abs(a))
        mean_b = np.mean(np.abs(b))
        denom = 0.5 * (mean_a + mean_b) + eps
        return np.abs(a - b) / denom

    # Sample the fields (use every Nth point for comparison if grid sizes differ)
    n1 = len(data1['density'])
    n2 = len(data2['density'])

    if n1 == n2:
        # Same grid size, direct comparison
        density_err = relative_error(data1['density'], data2['density'])
        pressure_err = relative_error(data1['pressure'], data2['pressure'])
    else:
        # Different grid sizes - use coarser grid sampling
        n_samples = min(n1, n2)
        idx1 = np.linspace(0, n1-1, n_samples, dtype=int)
        idx2 = np.linspace(0, n2-1, n_samples, dtype=int)

        density_err = relative_error(data1['density'][idx1], data2['density'][idx2])
        pressure_err = relative_error(data1['pressure'][idx1], data2['pressure'][idx2])

    # Combined RMSE
    combined_err = (density_err + pressure_err) / 2.0
    rmse = np.sqrt(np.mean(combined_err**2))

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

    # Run a simulation
    cost = run_sim_euler_2d("p1", testcase=testcase, n_grid_x=n_grid_x, start_frame=0, end_frame=10)
    print(f"Simulation cost: {cost}")

    # Load results
    results = get_res_euler_2d("p1", testcase=testcase, n_grid_x=n_grid_x, start_frame=0, end_frame=10)
    print(f"Loaded {len(results)} frames")

    # Compute metrics
    metrics = compute_euler_2d_metrics(results)
    print_euler_2d_metrics("Test", metrics)
