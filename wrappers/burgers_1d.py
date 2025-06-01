import os
import subprocess
import h5py
import numpy as np
import json
import matplotlib.pyplot as plt


def run_sim_burgers_1d(profile, cfl, k, w):
    """Run the burgers_1d simulation with the given parameters if not already simulated."""
    dir_path = f"sim_res/burgers_1d/{profile}_cfl_{cfl}_k_{k}_w_{w}/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"]

    # Run the simulation if not already done
    print(f"Running new simulation with parameters: cfl={cfl}, k={k}, w={w}")
    cmd = f"python costsci_tools/runners/burgers_1d.py --config-name={profile} cfl={cfl} k={k} w={w}"
    subprocess.run(cmd, shell=True, check=True)

    # Load the cost from the meta.json file
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]

    return cost


def get_res_burgers_1d(profile, cfl, k, w):
    """Load all time frames for a given parameter set, triggering a simulation if results are missing."""
    dir_path = f"sim_res/burgers_1d/{profile}_cfl_{cfl}_k_{k}_w_{w}/"
    results = []
    X = None

    # Check if at least one result file exists, otherwise trigger a simulation
    if not os.path.exists(dir_path) or not any(
        fname.startswith("res_") and fname.endswith(".h5") for fname in os.listdir(dir_path)
    ):
        print(f"No results found for parameters: cfl={cfl}, k={k}, w={w}. Triggering simulation.")
        run_sim_burgers_1d(profile, cfl, k, w)

    # Sort files by time frame
    files = [f for f in os.listdir(dir_path) if f.startswith("res_") and f.endswith(".h5")]
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        with h5py.File(file_path, "r") as f:
            results.append(np.array(f["u"]))
            if X is None:
                X = np.array(f["x"])

    return np.array(results), X


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


def compare_res_burgers_1d(profile1, cfl1, k1, w1, profile2, cfl2, k2, w2, linf_tolerance, rmse_tolerance):
    """Compare two sets of results using error norms and physical metrics.
    Returns:
        converged (bool): True if Linf and RMSE tolerances are met.
        metrics1 (dict): Metrics for case 1.
        metrics2 (dict): Metrics for case 2.
        linf_norm (float): Linfinity norm of difference.
        rmse (float): RMSE of difference.
    """
    res1, x1 = get_res_burgers_1d(profile1, cfl1, k1, w1)
    res2, x2 = get_res_burgers_1d(profile2, cfl2, k2, w2)

    # Error norms
    diff = np.abs(res1 - res2)
    linf_norm = np.max(diff)
    rmse = np.sqrt(np.mean(diff**2))

    # Conservation metrics
    metrics1 = compute_metrics(res1)
    metrics2 = compute_metrics(res2)

    converged = (
        linf_norm < linf_tolerance
        and rmse < rmse_tolerance
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

    print(f"Linf Norm: {linf_norm}")
    print(f"RMSE: {rmse}")

    return converged, metrics1, metrics2, linf_norm, rmse


if __name__ == "__main__":
    # Example usage
    k = 0
    w = 0.9
    profiles = ["p1"]
    # profiles = ["p1", "p2", "p3", "p4", "p5"]
    linf_tolerance = 0.02
    rmse_tolerance = 0.001

    # for profile in profiles:
    #     cfl_values = [1]
    #     linf_norms = []
    #     l2_norms = []
    #     converged = False

    #     while not converged:
    #         cfl1 = cfl_values[-1]
    #         cfl2 = cfl1 / 2
    #         cfl_values.append(cfl2)

    #         converged, metrics1, metrics2, linf_norm, rmse = compare_res_burgers_1d(
    #             profile, cfl1, k, w, profile, cfl2, k, w, linf_tolerance, rmse_tolerance
    #         )
    #         linf_norms.append(linf_norm)
    #         l2_norms.append(rmse)

    #     # Plotting for each profile
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(cfl_values[:-1], linf_norms, marker="o", label="Linf Norm")
    #     plt.plot(cfl_values[:-1], l2_norms, marker="s", label="RMSE")
    #     plt.xscale("log")
    #     plt.yscale("log")
    #     plt.xlabel("CFL")
    #     plt.ylabel("Norm")
    #     plt.title(f"Trajectory of Norms vs CFL for {profile}")
    #     plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    #     plt.legend()
    #     plt.savefig(f"norms_vs_cfl_{profile}.png")
    #     plt.close()

    _, _, _, linf_norm, rmse = compare_res_burgers_1d(
        "p1", 0.5, k, 1, "p1", 0.125, k, 1, linf_tolerance, rmse_tolerance
    )

    print(f"Difference in RMSE between 2nd CFL and converged CFL: {rmse}")
    print(f"Difference in Linf Norm between 2nd CFL and converged CFL: {linf_norm}")
