import os
import subprocess
import h5py
import numpy as np
import json

env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def run_sim_heat_1d(profile, cfl, n_space):
    """Run the heat1d simulation with the given CFL number."""
    cmd = f"python costsci_tools/runners/heat_1d.py  --config-name={profile} cfl={cfl} n_space={n_space}"
    subprocess.run(cmd, shell=True, check=True, env=env)

    dir_path = f"sim_res/heat_1d/{profile}_cfl_{cfl}_nx_{n_space}/"

    # get cost from the meta.json
    with open(os.path.join(dir_path, "meta.json"), "r") as f:
        meta = json.load(f)
        cost = meta["cost"]

    return cost


def get_res_heat_1d(profile, cfl, n_space):
    """Load all time frames for a given CFL number."""
    dir_path = f"sim_res/heat_1d/{profile}_cfl_{cfl}_nx_{n_space}/"
    results = []

    for t in range(25):
        file_path = os.path.join(dir_path, f"res_{t}.h5")
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

    return left_grad_diff < tolerance


# def interpolate_to_finer_grid(coarse_data, fine_data, coarse_x, fine_x):
#     """Interpolate data from coarse grid to fine grid using linear interpolation."""
#     interpolated_data = np.zeros_like(fine_data)

#     for t in range(coarse_data.shape[0]):
#         interpolated_data[t] = np.interp(fine_x, coarse_x, coarse_data[t])

#     return interpolated_data


# if __name__ == "__main__":
#     # Example usage
#     profile1 = "p1"
#     cfl1 = 0.5
#     n_space1 = 320

#     profile2 = "p1"
#     cfl2 = 0.5
#     n_space2 = 640

#     tolerance = 1e-5

#     res1, x1 = get_res_heat_1d(profile1, cfl1, n_space1)
#     res2, x2 = get_res_heat_1d(profile2, cfl2, n_space2)

#     # upsample res1 to match res2
#     res1_interp = interpolate_to_finer_grid(res1, res2, x1, x2)
#     # for each time step, compare the results
#     for t in range(res1.shape[0]):
#         diff = np.abs(res1_interp[t] - res2[t])
#         mean_diff = np.mean(diff)
#         print(f"Mean difference at time step {t}: {mean_diff}")
