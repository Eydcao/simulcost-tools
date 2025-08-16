import os
import subprocess
import h5py
import numpy as np
import json
from scipy.interpolate import RegularGridInterpolator


def run_sim_heat_steady_2d(profile, dx, relax, error_threshold, t_init):
    """Run the heat_steady_2d simulation with the given parameters if not already simulated."""
    dir_path = f"sim_res/heat_steady_2d/{profile}_dx_{dx}_relax_{relax}_Tinit_{t_init}_error_{error_threshold}/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"]

    # Run the simulation if not already done
    print(
        f"Running new simulation with parameters: dx={dx}, relax={relax}, error_threshold={error_threshold}, T_init={t_init}"
    )
    cmd = f"python runners/heat_steady_2d.py --config-name={profile} dx={dx} relax={relax} error_threshold={error_threshold} T_init={t_init}"
    subprocess.run(cmd, shell=True, check=True)

    # Load the cost from the meta.json file
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]

    return cost


def get_res_heat_steady_2d(profile, dx, relax, error_threshold, t_init):
    """Load final temperature field for given parameters, triggering a simulation if results are missing."""
    dir_path = f"sim_res/heat_steady_2d/{profile}_dx_{dx}_relax_{relax}_Tinit_{t_init}_error_{error_threshold}/"

    # Check if at least one result file exists, otherwise trigger a simulation
    if not os.path.exists(dir_path) or not any(
        fname.startswith("res_") and fname.endswith(".h5") for fname in os.listdir(dir_path)
    ):
        print(
            f"No results found for parameters: dx={dx}, relax={relax}, error_threshold={error_threshold}, T_init={t_init}. Triggering simulation."
        )
        run_sim_heat_steady_2d(profile, dx, relax, error_threshold, t_init)

    # Find the latest result file in the directory
    files = [f for f in os.listdir(dir_path) if f.startswith("res_") and f.endswith(".h5")]
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    latest_file = files[-1]

    file_path = os.path.join(dir_path, latest_file)
    with h5py.File(file_path, "r") as f:
        T = np.array(f["T"])
        X = np.array(f["x"])
        Y = np.array(f["y"])
        iter_count = np.array(f["iter"])

    return T, X, Y, iter_count


def compute_heat_steady_metrics(T, X, Y):
    """Compute physical metrics for steady-state heat transfer solution.
    Args:
        T: Temperature field (nx, ny)
        X: x coordinates
        Y: y coordinates
    Returns:
        Dictionary containing:
        - temperature_valid: bool if all temperatures are finite and within bounds
        - gradient_reasonable: bool if temperature gradients are reasonable
    """
    # Check for NaN/Infinity
    temperature_valid = np.all(np.isfinite(T))

    # Check temperature bounds (should be within boundary condition range)
    T_min = np.min([np.min(X), np.min(Y), 0.0])  # Assume 0 is minimum boundary
    T_max = np.max([np.max(X), np.max(Y), 1.0])  # Assume 1 is maximum boundary
    temperature_bounded = np.all((T >= T_min - 0.1) & (T <= T_max + 0.1))

    return {
        "temperature_valid": temperature_valid and temperature_bounded
    }


def print_heat_steady_metrics(name, metrics):
    """Print summary statistics for heat transfer metrics"""
    print(f"\n--- {name} Metrics ---")
    if not metrics:
        print("No metrics available (insufficient data)")
        return

    print(f"Temperature field valid: {metrics['temperature_valid']}")


def compare_res_heat_steady_2d(
    profile1, dx1, relax1, error_threshold1, t_init1, profile2, dx2, relax2, error_threshold2, t_init2, rmse_tolerance
):
    """Compare two sets of results using RMSE of temperature distribution on the middle vertical line (x=0.5).
    Returns:
        converged (bool): True if RMSE tolerance is met.
        metrics1 (dict): Metrics for case 1.
        metrics2 (dict): Metrics for case 2.
        rmse (float): RMSE of temperature difference.
    """
    res1, x1, y1, _ = get_res_heat_steady_2d(profile1, dx1, relax1, error_threshold1, t_init1)
    res2, x2, y2, _ = get_res_heat_steady_2d(profile2, dx2, relax2, error_threshold2, t_init2)

    # Find the index of x=0.5 in both datasets
    idx1 = np.argmin(np.abs(x1 - 0.5))
    idx2 = np.argmin(np.abs(x2 - 0.5))

    # Extract temperature values along the middle vertical line (x=0.5)
    T_line1 = res1[idx1, :]
    T_line2 = res2[idx2, :]

    # Interpolate T_line2 to match the y-coordinates of T_line1
    interpolator = RegularGridInterpolator((y2,), T_line2)
    T_line2_interp = interpolator(y1)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((T_line1 - T_line2_interp) ** 2))

    # Compute metrics
    metrics1 = compute_heat_steady_metrics(res1, x1, y1)
    metrics2 = compute_heat_steady_metrics(res2, x2, y2)

    # Convergence criteria
    converged = (
        rmse < rmse_tolerance
        and metrics1["temperature_valid"]
        and metrics2["temperature_valid"]
    )

    print_heat_steady_metrics("Case 1", metrics1)
    print_heat_steady_metrics("Case 2", metrics2)

    print(f"RMSE (middle line): {rmse}")

    return converged, metrics1, metrics2, rmse
