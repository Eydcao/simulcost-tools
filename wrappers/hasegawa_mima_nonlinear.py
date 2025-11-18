import os
import subprocess
import h5py
import numpy as np
import json
import sys
from scipy.interpolate import RegularGridInterpolator
from dotenv import load_dotenv
import yaml

# from solvers.utils import format_param_for_path

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


def _read_config_max_wall_time(profile):
    """
    Read max_wall_time from the config file for a given profile.

    Args:
        profile: Configuration profile name (e.g., "p1")

    Returns:
        max_wall_time value from config, or 120 if not found
    """
    # Try to find config file
    config_paths = [
        f"run_configs/hasegawa_mima_nonlinear/{profile}.yaml",
        f"costsci_tools/run_configs/hasegawa_mima_nonlinear/{profile}.yaml",
        f"../run_configs/hasegawa_mima_nonlinear/{profile}.yaml",
    ]

    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    max_wall_time = config.get('max_wall_time', 120)
                    return max_wall_time
            except Exception as e:
                print(f"Warning: Could not read config from {config_path}: {e}")
                continue

    # Fallback to default if config not found
    print(f"Warning: Could not find config for profile {profile}, using default max_wall_time=120")
    return 120


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


def get_results(profile, N, dt, max_wall_time=-1):
    """
    Get simulation results by running the simulation (if needed) and loading all frames.

    Args:
        profile: Configuration profile name
        N: Grid resolution
        dt: Time step size
        max_wall_time: Override for maximum wall time in seconds (default=-1).
                      - Default (-1): Use config default (typically 120s)
                      - None: Disable limit (no constraint) - saves to folder WITHOUT suffix
                      - Positive number: Override to that many seconds - saves to folder WITH suffix

    Returns:
        tuple: (cost, results_list, simulation_completed) where results_list contains all frame data
               and simulation_completed is True if wall time was NOT exceeded

    Note:
        Directory naming convention to avoid cache ambiguity:
        - Unconstrained (max_wall_time=None): {profile}_N_{N}_dt_{dt}/
        - Constrained: {profile}_N_{N}_dt_{dt}_wall_time_{value}/
    """
    # Determine the actual wall time value for directory naming
    # If max_wall_time is None (unconstrained), no suffix
    # Otherwise, we need the actual numeric value for the suffix
    if max_wall_time is None:
        # Unconstrained reference run - no suffix
        dir_suffix = ""
    elif max_wall_time == -1:
        # Use config default - read from config file
        actual_wall_time = _read_config_max_wall_time(profile)
        dir_suffix = f"_wall_time_{actual_wall_time}"
    else:
        # Explicit value provided
        actual_wall_time = max_wall_time
        dir_suffix = f"_wall_time_{actual_wall_time}"

    dir_path = _get_sim_path(f"sim_res/hasegawa_mima_nonlinear/{profile}_N_{N}_dt_{dt:.2e}{dir_suffix}/")
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                cost = meta["cost"]
            wall_time_total = meta.get("wall_time_total", 0.0)
            # Use wall_time_exceeded flag from metadata (more robust than hardcoded comparison)
            wall_time_exceeded = meta.get("wall_time_exceeded", False)
            simulation_completed = not wall_time_exceeded

            if not simulation_completed:
                print(f"Warning: Simulation did not complete (wall_time_exceeded=True, wall_time_total={wall_time_total:.1f}s)")
    else:
        # Run the simulation if not already done
        # Determine wall time description for logging
        if max_wall_time is None:
            wall_time_desc = "disabled"
        elif max_wall_time == -1:
            wall_time_desc = "config default"
        else:
            wall_time_desc = f"{max_wall_time}s"

        print(f"Running new nonlinear simulation with parameters: N={N}, dt={dt}, max_wall_time={wall_time_desc}")
        runner_path = _find_runner_path()

        # Build command
        if SIM_RES_BASE_DIR:
            dump_dir = os.path.join(SIM_RES_BASE_DIR, f"sim_res/hasegawa_mima_nonlinear/{profile}")
            cmd = f"{sys.executable} {runner_path} --config-name={profile} N={N} dt={dt} dump_dir={dump_dir}"
        else:
            cmd = f"{sys.executable} {runner_path} --config-name={profile} N={N} dt={dt}"

        # Add max_wall_time override if specified
        if max_wall_time is None:
            # Disable wall time limit
            cmd += " max_wall_time=null"
        elif max_wall_time > 0:
            # Override to specific value
            cmd += f" max_wall_time={max_wall_time}"
        # If max_wall_time == -1 (default), don't add anything - use config default

        subprocess.run(cmd, shell=True, check=True)

        # Read the meta.json to get cost and completion status
        with open(meta_path, "r") as f:
            meta = json.load(f)
            cost = meta["cost"]
            wall_time_total = meta.get("wall_time_total", 0.0)
            wall_time_exceeded = meta.get("wall_time_exceeded", False)
            simulation_completed = not wall_time_exceeded

            if not simulation_completed:
                print(f"Warning: Simulation did not complete (wall_time_exceeded=True, wall_time_total={wall_time_total:.1f}s)")

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

    return cost, results_list, simulation_completed


def compare_solutions(profile, params1, params2, tolerance_rmse, max_wall_time=120):
    """
    Compare two Hasegawa-Mima nonlinear simulations to check for convergence.
    Uses linear interpolation for different resolutions.

    Args:
        profile: Configuration profile name
        params1: Dictionary with first simulation parameters (coarse - to be evaluated)
        params2: Dictionary with second simulation parameters (fine - reference)
        tolerance_rmse: RMSE tolerance for convergence
        max_wall_time: Wall time limit for the coarse (evaluated) simulation (default: 120s)

    Returns:
        is_converged: Boolean indicating if solutions converged
        cost1: Cost for first simulation
        cost2: Cost for second simulation (or 0 if not run)
        rmse_diff: RMSE difference between solutions (or None if comparison not possible)

    Note:
        - Coarse simulation (params1): runs with wall time constraint (suffix: _wall_time_{value})
        - Fine simulation (params2): runs without constraint as reference (no suffix)
        - For N search: dt is scaled for reference to maintain CFL stability (dt ∝ 1/N)
    """
    # Run first (coarse) simulation WITH wall time constraint for evaluation
    cost1, results1, completed1 = get_results(profile, max_wall_time=max_wall_time, **params1)

    # Check if first simulation completed
    if not completed1:
        print(f"⚠️  First simulation hit wall time limit - skipping second simulation")
        print(f"   Params1: {params1}")
        return False, cost1, 0, None

    if not results1:
        print(f"⚠️  First simulation produced no results")
        return False, cost1, 0, None

    # Prepare params for reference simulation
    # If N values differ, scale dt for reference to maintain CFL stability (dt ∝ 1/N)
    params2_ref = params2.copy()
    if params1.get("N") != params2.get("N"):
        N1 = params1["N"]
        N2 = params2["N"]
        dt_proposal = params1["dt"]
        # CFL condition: dt ∝ 1/N, so dt_ref = dt_proposal * N1 / N2
        dt_ref = dt_proposal * N1 / N2
        params2_ref["dt"] = dt_ref
        print(f"   Scaling reference dt: {dt_proposal:.6e} → {dt_ref:.6e} (for CFL stability at N={N2})")

    # Run second (fine) simulation WITHOUT wall time limit (reference/ground truth)
    cost2, results2, completed2 = get_results(profile, max_wall_time=None, **params2_ref)

    # Reference should always complete (no wall time limit)
    if not results2:
        print(f"⚠️  Reference simulation produced no results")
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
