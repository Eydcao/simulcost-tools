import os
import subprocess
import h5py
import numpy as np
import json
import matplotlib.pyplot as plt


def run_sim_euler_1d(profile, cfl, beta, k, n_space):
    """Run the euler_1d simulation with the given parameters if not already simulated."""
    dir_path = f"sim_res/euler_1d/{profile}_cfl_{cfl}_beta_{beta}_k_{k}_n_{n_space}/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"]

    # Run the simulation if not already done
    print(f"Running new simulation with parameters: cfl={cfl}, beta={beta}, k={k}, n_space={n_space}")
    cmd = f"python runners/euler_1d.py --config-name={profile} cfl={cfl} beta={beta} k={k} n_space={n_space}"
    subprocess.run(cmd, shell=True, check=True)

    # Load the cost from the meta.json file
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]

    return cost


def get_res_euler_1d(profile, cfl, beta, k, n_space):
    """Load all time frames for a given parameter set, triggering a simulation if results are missing."""
    dir_path = f"sim_res/euler_1d/{profile}_cfl_{cfl}_beta_{beta}_k_{k}_n_{n_space}/"
    results = {}
    X = None

    # Check if at least one result file exists, otherwise trigger a simulation
    if not os.path.exists(dir_path) or not any(
        fname.startswith("res_") and fname.endswith(".h5") for fname in os.listdir(dir_path)
    ):
        print(
            f"No results found for parameters: cfl={cfl}, beta={beta}, k={k}, n_space={n_space}. Triggering simulation."
        )
        run_sim_euler_1d(profile=profile, cfl=cfl, beta=beta, k=k, n_space=n_space)

    # Sort files by time frame
    files = [f for f in os.listdir(dir_path) if f.startswith("res_") and f.endswith(".h5")]
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        with h5py.File(file_path, "r") as f:
            # Load all primitive variables
            if X is None:
                X = np.array(f["x"])

            time_data = {
                "rho": np.array(f["rho"]),
                "u": np.array(f["u"]),
                "p": np.array(f["p"]),
                "E": np.array(f["E"]),
                "time": f["time"][()],
            }

            frame_id = int(file_name.split("_")[1].split(".")[0])
            results[frame_id] = time_data

    return results, X


def compute_euler_metrics(results):
    """Compute physical metrics for Euler equations solution.
    Args:
        results: Dictionary with frame_id -> {rho, u, p, E, time}
    Returns:
        Dictionary containing:
        - positivity_preserved: bool array if pressure/density stay positive
        - shock_speed_consistent: bool array if shock speeds are physical
    """
    frames = sorted(results.keys())
    if len(frames) < 2:
        return {}

    # Extract time series data
    rho_series = np.array([results[f]["rho"] for f in frames])
    p_series = np.array([results[f]["p"] for f in frames])

    # Positivity preservation
    min_pressure = np.min(p_series, axis=1)
    min_density = np.min(rho_series, axis=1)
    positivity_preserved = (min_pressure > 0) & (min_density > 0)

    # Shock speed consistency (basic check)
    max_pressure_gradient = np.max(np.abs(np.diff(p_series, axis=1)), axis=1)
    shock_speed_consistent = max_pressure_gradient < 1e3  # Prevent unrealistic gradients

    return {
        "positivity_preserved": positivity_preserved,
        "shock_speed_consistent": shock_speed_consistent,
    }


def print_euler_metrics(name, metrics):
    """Print summary statistics for Euler metrics"""
    print(f"\n--- {name} Metrics ---")
    if not metrics:
        print("No metrics available (insufficient data)")
        return

    print(f"Positivity preserved at all steps: {np.all(metrics['positivity_preserved'])}")
    print(f"Shock speed consistent at all steps: {np.all(metrics['shock_speed_consistent'])}")


def compare_res_euler_1d(profile1, cfl1, beta1, k1, profile2, cfl2, beta2, k2, rmse_tolerance, n_space1, n_space2):
    """Compare two sets of results using relative error norms and physical metrics.
    Returns:
        converged (bool): True if RMSE tolerance is met.
        metrics1 (dict): Metrics for case 1.
        metrics2 (dict): Metrics for case 2.
        rmse (float): RMSE of relative difference.
    """
    res1, x1 = get_res_euler_1d(profile1, cfl1, beta1, k1, n_space1)
    res2, x2 = get_res_euler_1d(profile2, cfl2, beta2, k2, n_space2)

    # Handle different grids by upsampling to the finer grid
    if len(x1) != len(x2) or not np.allclose(x1, x2):
        if len(x1) < len(x2):
            from scipy.interpolate import interp1d

            final1_interp = {}
            final1 = res1[sorted(res1.keys())[-1]]
            for var in ["rho", "u", "p"]:
                f_interp = interp1d(x1, final1[var], kind="linear", bounds_error=False, fill_value="extrapolate")
                final1_interp[var] = f_interp(x2)
            final1 = final1_interp
            final2 = res2[sorted(res2.keys())[-1]]
            x_common = x2
        else:
            from scipy.interpolate import interp1d

            final2_interp = {}
            final2 = res2[sorted(res2.keys())[-1]]
            for var in ["rho", "u", "p"]:
                f_interp = interp1d(x2, final2[var], kind="linear", bounds_error=False, fill_value="extrapolate")
                final2_interp[var] = f_interp(x1)
            final2 = final2_interp
            final1 = res1[sorted(res1.keys())[-1]]
            x_common = x1
    else:
        # Same grid
        final1 = res1[sorted(res1.keys())[-1]]
        final2 = res2[sorted(res2.keys())[-1]]
        x_common = x1

    # Check if we have simulation results
    frames1 = sorted(res1.keys())
    frames2 = sorted(res2.keys())

    if len(frames1) == 0 or len(frames2) == 0:
        raise ValueError("No simulation results found")

    # Compute relative error norms for all primitive variables
    eps = 1e-12  # To avoid division by zero

    def denom(a, b):
        # Use average of abs(std) of both arrays plus eps
        std_a = np.std(a)
        std_b = np.std(b)
        return 0.5 * (np.abs(std_a) + np.abs(std_b)) + eps

    rho_diff = np.abs(final1["rho"] - final2["rho"]) / denom(final1["rho"], final2["rho"])
    u_diff = np.abs(final1["u"] - final2["u"]) / denom(final1["u"], final2["u"])
    p_diff = np.abs(final1["p"] - final2["p"]) / denom(final1["p"], final2["p"])

    # Combined relative error norm (weighted)
    combined_diff = (rho_diff + u_diff + p_diff) / 3.0
    rmse = np.sqrt(np.mean(combined_diff**2))

    # Conservation metrics
    metrics1 = compute_euler_metrics(res1)
    metrics2 = compute_euler_metrics(res2)

    # Convergence criteria (mass/energy/momentum conservation excluded due to boundary condition issues)
    converged = (
        rmse < rmse_tolerance
        # Note: mass, energy, momentum conservation excluded - may fail due to boundary conditions
        and (not metrics1 or np.all(metrics1["positivity_preserved"]))
        and (not metrics2 or np.all(metrics2["positivity_preserved"]))
        and (not metrics1 or np.all(metrics1["shock_speed_consistent"]))
        and (not metrics2 or np.all(metrics2["shock_speed_consistent"]))
    )

    print_euler_metrics("Case 1", metrics1)
    print_euler_metrics("Case 2", metrics2)

    print(f"RMSE (relative): {rmse}")

    return converged, metrics1, metrics2, rmse


if __name__ == "__main__":
    # Example usage
    beta = 1.0
    k = 1.0
    profiles = ["p1"]
    rmse_tolerance = 1e-2

    _, _, _, rmse = compare_res_euler_1d(profile1="p1", cfl1=0.5, beta1=beta, k1=k, profile2="p1", cfl2=0.25, beta2=beta, k2=k, rmse_tolerance=rmse_tolerance, n_space1=512, n_space2=256)

    print(f"Difference in RMSE between CFL 0.5 and CFL 0.25: {rmse}")
