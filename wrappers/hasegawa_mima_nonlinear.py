import os
import subprocess
import h5py
import numpy as np
import json
import sys
from scipy.interpolate import RegularGridInterpolator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get base directory for simulation results from environment variable
# If not set, use current directory (maintains backward compatibility)
SIM_RES_BASE_DIR = os.getenv("SIM_RES_BASE_DIR", None)
if SIM_RES_BASE_DIR:
    print(f"✅ Using custom simulation results directory: {SIM_RES_BASE_DIR}")


def _get_sim_path(relative_path):
    """Construct simulation path, using absolute path if SIM_RES_BASE_DIR is set."""
    if SIM_RES_BASE_DIR:
        return os.path.join(SIM_RES_BASE_DIR, relative_path)
    return relative_path


def format_param_for_path(value):
    """
    Format parameter values for clean folder/file names.

    Args:
        value: Parameter value (float, int, or other)

    Returns:
        str: Cleanly formatted string suitable for file paths
    """
    if isinstance(value, float):
        if value >= 1e-3 and value < 1e3:
            # Use fixed point for reasonable range, remove trailing zeros
            return f"{value:.6g}".rstrip("0").rstrip(".")
        else:
            # Use scientific notation for very small/large values
            return f"{value:.2e}"
    else:
        return str(value)


def _find_runner_path():
    """Automatically find the correct path to hasegawa_mima_nonlinear.py runner."""
    # Get current working directory
    cwd = os.getcwd()

    # List of possible runner paths relative to different working directories
    possible_paths = []

    # If working from project root (SimulCost-Bench/)
    if cwd.endswith("SimulCost-Bench"):
        possible_paths.extend(
            ["costsci_tools/runners/hasegawa_mima_nonlinear.py", "runners/hasegawa_mima_nonlinear.py"]
        )
    # If working from costsci_tools/ subdirectory
    elif cwd.endswith("costsci_tools") or "costsci_tools" in cwd:
        possible_paths.extend(
            [
                "runners/hasegawa_mima_nonlinear.py",
                "../runners/hasegawa_mima_nonlinear.py",
                "costsci_tools/runners/hasegawa_mima_nonlinear.py",
            ]
        )

    # Add generic fallback paths
    possible_paths.extend(
        [
            "runners/hasegawa_mima_nonlinear.py",
            "costsci_tools/runners/hasegawa_mima_nonlinear.py",
            "./runners/hasegawa_mima_nonlinear.py",
            "../runners/hasegawa_mima_nonlinear.py",
            "../../runners/hasegawa_mima_nonlinear.py",
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

    # If none found, raise an error with helpful information
    raise FileNotFoundError(
        f"Could not find hasegawa_mima_nonlinear.py runner in any expected location.\n"
        f"Current working directory: {cwd}\n"
        f"Searched paths: {unique_paths}\n"
        f"Please ensure the runner exists or update the search paths."
    )


def get_results(profile, N, dt):
    """
    Get simulation results by running the simulation (if needed) and loading all frames.

    Args:
        profile: Configuration profile name
        N: Grid resolution
        dt: Time step size

    Returns:
        tuple: (cost, results_list) where results_list contains all frame data
    """
    dir_path = _get_sim_path(f"sim_res/hasegawa_mima_nonlinear/{profile}_N_{N}_dt_{dt:.2e}_nonlinear/")
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                cost = meta["cost"]
    else:
        # Run the simulation if not already done
        print(f"Running new nonlinear simulation with parameters: N={N}, dt={dt}")
        runner_path = _find_runner_path()
        if SIM_RES_BASE_DIR:
            dump_dir = os.path.join(SIM_RES_BASE_DIR, f"sim_res/hasegawa_mima_nonlinear/{profile}")
            cmd = f"{sys.executable} {runner_path} --config-name={profile} N={N} dt={dt} dump_dir={dump_dir}"
        else:
            cmd = f"{sys.executable} {runner_path} --config-name={profile} N={N} dt={dt}"
        subprocess.run(cmd, shell=True, check=True)

        # Read the meta.json to get cost
        with open(meta_path, "r") as f:
            meta = json.load(f)
            cost = meta["cost"]

    # Load all frames
    results_list = []
    frame_files = sorted([f for f in os.listdir(dir_path) if f.startswith("frame_") and f.endswith(".h5")])

    for frame_file in frame_files:
        frame_num = int(frame_file.split("_")[1].split(".")[0])
        h5_file = os.path.join(dir_path, frame_file)

        with h5py.File(h5_file, "r") as f:
            result = {
                "phi": f["phi"][:],
                "coordinates_x": f["coordinates_x"][:],
                "coordinates_y": f["coordinates_y"][:],
                "time": f.attrs["time"],
                "N": f.attrs["N"],
                "dt": f.attrs["dt"],
                "dealias_ratio": f.attrs["dealias_ratio"],
            }
            results_list.append(result)

    return cost, results_list


def compare_solutions(profile, params1, params2, tolerance_rmse):
    """
    Compare two Hasegawa-Mima nonlinear simulations to check for convergence.
    Uses linear interpolation for different resolutions.

    Args:
        profile: Configuration profile name
        params1: Dictionary with first simulation parameters (coarse)
        params2: Dictionary with second simulation parameters (fine)
        tolerance_rmse: RMSE tolerance for convergence

    Returns:
        is_converged: Boolean indicating if solutions converged
        cost1: Cost for first simulation
        cost2: Cost for second simulation
        rmse_diff: RMSE difference between solutions
    """
    # Get both simulation results
    cost1, results1 = get_results(profile, **params1)
    cost2, results2 = get_results(profile, **params2)

    if not results1 or not results2:
        return False, cost1, cost2, None

    # Calculate L2 error between resolutions for all frames using interpolation
    l2_errors = []

    for res1, res2 in zip(results1, results2):
        # Get phi fields and coordinates
        phi1 = np.array(res1["phi"])
        phi2 = np.array(res2["phi"])
        x1 = np.array(res1["coordinates_x"])
        y1 = np.array(res1["coordinates_y"])
        x2 = np.array(res2["coordinates_x"])
        y2 = np.array(res2["coordinates_y"])
        N1 = phi1.shape[0]
        N2 = phi2.shape[0]

        # Check for NaN or Inf
        if not (np.all(np.isfinite(phi1)) and np.all(np.isfinite(phi2))):
            print(f"Warning: NaN or Inf detected in solutions at time {res1['time']}")
            return False, cost1, cost2, None

        # Interpolate lower resolution to higher resolution
        if N1 < N2:
            # Interpolate phi1 (coarse) to phi2's grid (fine)
            interpolator = RegularGridInterpolator((x1, y1), phi1, method="linear", bounds_error=False, fill_value=None)
            X2, Y2 = np.meshgrid(x2, y2, indexing="ij")
            phi1_interp = interpolator(np.stack([X2.ravel(), Y2.ravel()], axis=-1)).reshape(N2, N2)
            phi_coarse_on_fine = phi1_interp
            phi_fine = phi2
        elif N2 < N1:
            # Interpolate phi2 (coarse) to phi1's grid (fine)
            interpolator = RegularGridInterpolator((x2, y2), phi2, method="linear", bounds_error=False, fill_value=None)
            X1, Y1 = np.meshgrid(x1, y1, indexing="ij")
            phi2_interp = interpolator(np.stack([X1.ravel(), Y1.ravel()], axis=-1)).reshape(N1, N1)
            phi_coarse_on_fine = phi2_interp
            phi_fine = phi1
        else:
            # Same resolution, no interpolation needed
            phi_coarse_on_fine = phi1
            phi_fine = phi2

        # Calculate L2 error
        diff = phi_fine - phi_coarse_on_fine
        l2_error = np.sqrt(np.mean(diff**2))
        l2_errors.append(l2_error)

    # Use mean L2 error across all frames
    rmse_diff = np.mean(l2_errors)

    # Check if error between resolutions is below tolerance
    is_converged = rmse_diff <= tolerance_rmse

    return is_converged, cost1, cost2, rmse_diff
