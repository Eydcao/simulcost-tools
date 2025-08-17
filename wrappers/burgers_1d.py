import os
import subprocess
import h5py
import numpy as np
import json
import matplotlib.pyplot as plt


def run_sim_burgers_1d(profile, cfl, k, beta, n_space):
    """Run the burgers_1d simulation with the given parameters if not already simulated."""
    dir_path = f"sim_res/burgers_1d/{profile}_cfl_{cfl}_k_{k}_beta_{beta}_n_{n_space}/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"]

    # Run the simulation if not already done
    print(f"Running new simulation with parameters: cfl={cfl}, k={k}, beta={beta}, n_space={n_space}")
    cmd = f"python runners/burgers_1d.py --config-name={profile} cfl={cfl} k={k} beta={beta} n_space={n_space}"
    subprocess.run(cmd, shell=True, check=True)

    # Load the cost from the meta.json file
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]

    return cost


def get_res_burgers_1d(profile, cfl, k, beta, n_space):
    """Load all time frames for a given parameter set, triggering a simulation if results are missing."""
    dir_path = f"sim_res/burgers_1d/{profile}_cfl_{cfl}_k_{k}_beta_{beta}_n_{n_space}/"
    results = {}
    X = None

    # Check if at least one result file exists, otherwise trigger a simulation
    if not os.path.exists(dir_path) or not any(
        fname.startswith("res_") and fname.endswith(".h5") for fname in os.listdir(dir_path)
    ):
        print(
            f"No results found for parameters: cfl={cfl}, k={k}, beta={beta}, n_space={n_space}. Triggering simulation."
        )
        run_sim_burgers_1d(profile, cfl, k, beta, n_space)

    # Sort files by time frame
    files = [f for f in os.listdir(dir_path) if f.startswith("res_") and f.endswith(".h5")]
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        frame_number = int(file_name.split("_")[1].split(".")[0])
        with h5py.File(file_path, "r") as f:
            results[frame_number] = np.array(f["u"])
            if X is None:
                X = np.array(f["x"])

    return results, X


def compute_metrics(u):
    """Compute physical metrics for Burgers' equation solution.
    Args:
        u: np.ndarray with shape [nt, nx] (time steps, spatial points)
    Returns:
        Dictionary containing:
        - mass_conserved: bool array if mass is conserved (shape [nt-1])
        - energy_non_increasing: bool array if energy doesn't increase (shape [nt-1])
        - TV_non_increasing: bool array if TV doesn't increase (shape [nt-1])
        - max_principle_satisfied: bool array if satisfied (shape [nt])
    """

    # Mass (mean value)
    mass = np.mean(u, axis=1)
    mass_conserved = np.isclose(mass[1:], mass[0], rtol=1e-3)

    # Energy (mean squared)
    energy = np.mean(u**2, axis=1)
    energy_non_increasing = np.diff(energy) <= 1e-3  # Allow small numerical increases

    # Total Variation (mean absolute difference)
    TV = np.mean(np.abs(np.diff(u, axis=1)), axis=1)
    TV_non_increasing = np.diff(TV) <= 1e-3

    # Maximum principle
    initial_max = np.max(u[0])
    max_principle_violation = np.maximum(u[1:] - initial_max, 0)
    max_violation = np.max(max_principle_violation, axis=1)
    max_principle_satisfied = max_violation <= 1e-3

    return {
        "mass_conserved": mass_conserved,
        "energy_non_increasing": energy_non_increasing,
        "TV_non_increasing": TV_non_increasing,
        "max_principle_satisfied": max_principle_satisfied,
    }


# Print summary statistics
def print_metrics(name, metrics):
    print(f"\n--- {name} Metrics ---")
    print(f"Mass conserved at all steps: {np.all(metrics['mass_conserved'])}")
    print(f"Energy non-increasing at all steps: {np.all(metrics['energy_non_increasing'])}")
    print(f"TV non-increasing at all steps: {np.all(metrics['TV_non_increasing'])}")
    print(f"Max principle satisfied at all steps: {np.all(metrics['max_principle_satisfied'])}")


def compare_res_burgers_1d(profile1, cfl1, k1, beta1, profile2, cfl2, k2, beta2, rmse_tolerance, n_space1, n_space2):
    """Compare two sets of results using error norms and physical metrics.
    Returns:
        converged (bool): True if RMSE tolerance is met.
        metrics1 (dict): Metrics for case 1.
        metrics2 (dict): Metrics for case 2.
        rmse (float): RMSE of difference.
    """
    res1_dict, x1 = get_res_burgers_1d(profile1, cfl1, k1, beta1, n_space1)
    res2_dict, x2 = get_res_burgers_1d(profile2, cfl2, k2, beta2, n_space2)

    # Convert dictionary results to arrays for comparison
    frames1 = sorted(res1_dict.keys())
    frames2 = sorted(res2_dict.keys())
    res1 = np.array([res1_dict[frame] for frame in frames1])
    res2 = np.array([res2_dict[frame] for frame in frames2])

    # For different n_space values, we need to interpolate to the same grid
    if len(x1) != len(x2):
        # Interpolate to the finer grid
        if len(x2) > len(x1):
            # Interpolate res1 to x2
            res1_interp = []
            for i in range(res1.shape[0]):
                res1_interp.append(np.interp(x2, x1, res1[i]))
            res1 = np.array(res1_interp)
        else:
            # Interpolate res2 to x1
            res2_interp = []
            for i in range(res2.shape[0]):
                res2_interp.append(np.interp(x1, x2, res2[i]))
            res2 = np.array(res2_interp)

    # Error norms
    eps = 1e-12  # To avoid division by zero

    def denom(a, b):
        # Use average of abs(std) of both arrays plus eps
        std_a = np.std(a)
        std_b = np.std(b)
        return 0.5 * (np.abs(std_a) + np.abs(std_b)) + eps

    diff = np.abs(res1 - res2) / denom(res1, res2)
    rmse = np.sqrt(np.mean(diff**2))

    # Conservation metrics
    metrics1 = compute_metrics(res1)
    metrics2 = compute_metrics(res2)

    converged = (
        rmse < rmse_tolerance
        and np.all(metrics1["mass_conserved"])
        and np.all(metrics2["mass_conserved"])
        and np.all(metrics1["energy_non_increasing"])
        and np.all(metrics2["energy_non_increasing"])
        and np.all(metrics1["TV_non_increasing"])
        and np.all(metrics2["TV_non_increasing"])
        and np.all(metrics1["max_principle_satisfied"])
        and np.all(metrics2["max_principle_satisfied"])
    )

    print_metrics("Case 1", metrics1)
    print_metrics("Case 2", metrics2)

    # print(f"Linf Norm (relative): {linf_norm}")
    print(f"RMSE (relative): {rmse}")

    return converged, metrics1, metrics2, rmse
