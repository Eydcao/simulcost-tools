import os
import subprocess
import h5py
import numpy as np
import json

env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _find_runner_path():
    """Automatically find the correct path to heat_1d.py runner."""
    # Get current working directory
    cwd = os.getcwd()
    
    # List of possible runner paths relative to different working directories
    possible_paths = []
    
    # If working from project root (SimulCost-Bench/)
    if cwd.endswith('SimulCost-Bench'):
        possible_paths.extend([
            "costsci_tools/runners/heat_1d.py",
            "runners/heat_1d.py"
        ])
    # If working from costsci_tools/ subdirectory
    elif cwd.endswith('costsci_tools') or 'costsci_tools' in cwd:
        possible_paths.extend([
            "runners/heat_1d.py",
            "../runners/heat_1d.py",
            "costsci_tools/runners/heat_1d.py"
        ])
    
    # Add generic fallback paths
    possible_paths.extend([
        "runners/heat_1d.py",
        "costsci_tools/runners/heat_1d.py",
        "./runners/heat_1d.py",
        "../runners/heat_1d.py",
        "../../runners/heat_1d.py"
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
    
    # If none found, raise an error with helpful information
    raise FileNotFoundError(
        f"Could not find heat_1d.py runner in any expected location.\n"
        f"Current working directory: {cwd}\n"
        f"Searched paths: {unique_paths}\n"
        f"Please ensure the runner exists or update the search paths."
    )


def run_sim_heat_1d(profile, cfl, n_space):
    """Run the heat1d simulation with the given CFL number if not already simulated."""
    dir_path = f"sim_res/heat_1d/{profile}_cfl_{cfl}_nx_{n_space}/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                return meta["cost"]

    # Run the simulation if not already done
    runner_path = _find_runner_path()
    cmd = f"python {runner_path} --config-name={profile} cfl={cfl} n_space={n_space}"
    subprocess.run(cmd, shell=True, check=True, env=env)

    # Load the cost from the meta.json file
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]

    return cost


def get_res_heat_1d(profile, cfl, n_space):
    """Load all time frames for a given CFL number, triggering a run if files are missing."""
    dir_path = f"sim_res/heat_1d/{profile}_cfl_{cfl}_nx_{n_space}/"
    results = []

    # Check if the first result file exists, trigger a run if not
    file_path = os.path.join(dir_path, "res_0.h5")
    if not os.path.exists(file_path):
        run_sim_heat_1d(profile, cfl, n_space)

    # Sort files by time frame
    files = [f for f in os.listdir(dir_path) if f.startswith("res_") and f.endswith(".h5")]
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        with h5py.File(file_path, "r") as f:
            results.append(np.array(f["T"]))
            X = np.array(f["x"])

    return np.array(results), X


def compare_res_heat_1d(profile1, cfl1, n_space1, profile2, cfl2, n_space2, tolerance):
    """Compare two sets of results with potential grid size mismatch."""
    res1, x1 = get_res_heat_1d(profile1, cfl1, n_space1)
    res2, x2 = get_res_heat_1d(profile2, cfl2, n_space2)

    left_temp_grad1 = (res1[-1, 1] - res1[-1, 0]) / (x1[1] - x1[0])
    left_grad_grad2 = (res2[-1, 1] - res2[-1, 0]) / (x2[1] - x2[0])

    left_grad_diff = (
        np.abs(left_temp_grad1 - left_grad_grad2) / (np.abs(left_grad_grad2) + np.abs(left_temp_grad1)) * 2.0
    )

    print(f"Left boundary temperature gradient 1: {left_temp_grad1}")
    print(f"Left boundary temperature gradient 2: {left_grad_grad2}")
    print(f"Relative left boundary gradient difference: {left_grad_diff}")

    return left_grad_diff < tolerance, left_grad_diff


# def interpolate_to_finer_grid(coarse_data, fine_data, coarse_x, fine_x):
#     """Interpolate data from coarse grid to fine grid using linear interpolation."""
#     interpolated_data = np.zeros_like(fine_data)

#     for t in range(coarse_data.shape[0]):
#         interpolated_data[t] = np.interp(fine_x, coarse_x, coarse_data[t])

#     return interpolated_data


if __name__ == "__main__":
    # Example usage
    profile1 = "p8"
    cfl1 = 0.25
    n_space1 = 75

    profile2 = "p8"
    cfl2 = 0.25
    n_space2 = 150

    tolerance = 0.01

    res1, x1 = get_res_heat_1d(profile1, cfl1, n_space1)
    res2, x2 = get_res_heat_1d(profile2, cfl2, n_space2)

    print(compare_res_heat_1d(profile1, cfl1, n_space1, profile2, cfl2, n_space2, tolerance))

    # upsample res1 to match res2
    # res1_interp = interpolate_to_finer_grid(res1, res2, x1, x2)
    # for each time step, compare the results
    # for t in range(res1.shape[0]):
    #     diff = np.abs(res1_interp[t] - res2[t])
    #     mean_diff = np.mean(diff)
    #     print(f"Mean difference at time step {t}: {mean_diff}")
