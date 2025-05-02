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
    Compare two sets of results using the RMSE of the temperature distribution on the middle vertical line (x=0.5).
    """
    res1, x1, y1, _ = get_res_heat_steady_2d(profile1, dx1, relax1, error_threshold1, t_init1)
    res2, x2, y2, _ = get_res_heat_steady_2d(profile2, dx2, relax2, error_threshold2, t_init2)

    # Find the index of x=0.5 in both datasets
    idx1 = np.argmin(np.abs(x1 - 0.5))
    idx2 = np.argmin(np.abs(x2 - 0.5))

    # Extract temperature values along the middle vertical line (x=0.5)
    T_line1 = res1[:, idx1]
    T_line2 = res2[:, idx2]

    # # save to fig plot for debugging
    # import matplotlib.pyplot as plt

    # plt.plot(y1, T_line1, label=f"dx={dx1}")
    # plt.plot(y2, T_line2, label=f"dx={dx2}")
    # plt.xlabel("y")
    # plt.ylabel("Temperature")
    # plt.title("Temperature Distribution at x=0.5")
    # plt.legend()
    # plt.savefig(f"temp_dist_x_0.5_dx_{dx1}_{dx2}.png")
    # plt.close()

    # Interpolate T_line2 to match the y-coordinates of T_line1
    interpolator = RegularGridInterpolator((y2,), T_line2)
    T_line2_interp = interpolator(y1)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((T_line1 - T_line2_interp) ** 2))

    print(f"RMSE of temperature distribution on x=0.5: {rmse:.6f}")
    print(f"Tolerance: {tolerance:.6f}")

    return rmse < tolerance


if __name__ == "__main__":
    # Example usage: compare dx  0.01 and 0.005
    profile = "p1"
    dx = 0.01
    relax = 1.0
    error_threshold = 1e-7
    t_init = 0.25

    tolerance = 1e-6

    # Compare results
    is_converged = compare_res_heat_steady_2d(
        profile, dx, relax, error_threshold, t_init, profile, dx / 2, relax, error_threshold, t_init, tolerance
    )
    print(f"Convergence achieved: {is_converged}")
