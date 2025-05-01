import os
import subprocess
import h5py
import numpy as np
import json
from scipy.interpolate import RegularGridInterpolator


def run_sim_heat_steady_2d(profile, dx, relax, error_threshold, t_init):
    """Run the heat_steady_2d simulation with the given parameters."""
    cmd = f"python runners/heat_steady_2d.py --config-name={profile} dx={dx} relax={relax} error_threshold={error_threshold} T_init={t_init}"
    subprocess.run(cmd, shell=True, check=True)

    dir_path = f"sim_res/heat_steady_2d/{profile}_dx{dx}_relax_{relax}_Tinit_{t_init}_error_{error_threshold}/"

    # get cost from the meta.json
    with open(os.path.join(dir_path, "meta.json"), "r") as f:
        meta = json.load(f)
        cost = meta["cost"]

    return cost


def get_res_heat_steady_2d(profile, dx, relax, error_threshold, t_init):
    """Load final temperature field for given parameters."""
    dir_path = f"sim_res/heat_steady_2d/{profile}_dx{dx}_relax_{relax}_Tinit_{t_init}_error_{error_threshold}/"

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


def compare_res_heat_steady_2d(
    profile1, dx1, relax1, error_threshold1, t_init1, profile2, dx2, relax2, error_threshold2, t_init2, tolerance
):
    """
    Compare two sets of results using heat flux at top boundary (excluding corners) as metric.
    Heat flux is proportional to temperature gradient in y-direction.
    """
    res1, x1, y1, _ = get_res_heat_steady_2d(profile1, dx1, relax1, error_threshold1, t_init1)
    res2, x2, y2, _ = get_res_heat_steady_2d(profile2, dx2, relax2, error_threshold2, t_init2)

    # Calculate gradient at top boundary: (T_top - T_second) / dy
    # Exclude the corners (first and last columns)
    grad1 = (res1[1:-1, -1] - res1[1:-1, -2]) / dx1
    sum_grad1 = np.mean(grad1)

    # For res2: Extract second-to-top row to calculate gradient with top boundary
    grad2 = (res2[1:-1, -1] - res2[1:-1, -2]) / dx2
    sum_grad2 = np.mean(grad2)

    # Calculate relative difference
    avg_sum_grad = (sum_grad1 + sum_grad2) / 2 + 1e-10
    rel_diff = np.abs(sum_grad1 - sum_grad2) / avg_sum_grad

    print(f"Sum of gradients for dx={dx1}: {sum_grad1:.6f}")
    print(f"Sum of gradients for dx={dx2}: {sum_grad2:.6f}")
    print(f"Relative heat flux difference: {rel_diff:.6f}")
    print(f"Tolerance: {tolerance:.6f}")

    return rel_diff < tolerance


if __name__ == "__main__":
    # Example usage: compare dx  0.01 and 0.005
    profile = "p1"
    dx = 0.0025
    relax = 1.0
    error_threshold = 1e-7
    t_init = 0.25

    tolerance = 1e-3

    # Compare results
    is_converged = compare_res_heat_steady_2d(
        profile, dx, relax, error_threshold, t_init, profile, dx / 2, relax, error_threshold, t_init, tolerance
    )
    print(f"Convergence achieved: {is_converged}")
