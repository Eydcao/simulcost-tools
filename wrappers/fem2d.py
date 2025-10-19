import os
import subprocess
import numpy as np
import json
import sys
from pathlib import Path


def _find_runner_path():
    """Automatically find the correct path to fem2d.py runner."""
    # Get current working directory
    cwd = os.getcwd()

    # List of possible runner paths relative to different working directories
    possible_paths = []

    # If working from project root (SimulCost-Bench/)
    if cwd.endswith('SimulCost-Bench'):
        possible_paths.extend([
            "costsci_tools/runners/fem2d.py",
            "runners/fem2d.py"
        ])
    # If working from costsci_tools/ subdirectory
    elif cwd.endswith('costsci_tools') or 'costsci_tools' in cwd:
        possible_paths.extend([
            "runners/fem2d.py",
            "../runners/fem2d.py",
            "costsci_tools/runners/fem2d.py"
        ])

    # Add generic fallback paths
    possible_paths.extend([
        "runners/fem2d.py",
        "costsci_tools/runners/fem2d.py",
        "./runners/fem2d.py",
        "../runners/fem2d.py",
        "../../runners/fem2d.py"
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
        f"Could not find fem2d.py runner in any expected location.\n"
        f"Current working directory: {cwd}\n"
        f"Searched paths: {unique_paths}\n"
        f"Please ensure the runner exists or update the search paths."
    )


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


def run_sim_fem2d(profile, nx, dt, newton_v_res_tol):
    """Run the fem2d simulation with the given parameters if not already simulated.

    Args:
        profile: Profile name (p1, p2, p3, etc.)
        nx: Grid resolution parameter
        dt: Time step size
        newton_v_res_tol: Newton velocity residual tolerance (convergence criterion)

    Returns:
        tuple: (cost, is_converged)
    """
    # Create directory path based on parameters (matching solver format)
    dir_path = f"sim_res/fem2d/{profile}_nx{nx}_dt{format_param_for_path(dt)}_nvrestol{format_param_for_path(newton_v_res_tol)}/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta and "is_converged" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"], meta["is_converged"]

    # Run the simulation if not already done
    print(f"Running new simulation with parameters: nx={nx}, dt={dt}, newton_v_res_tol={newton_v_res_tol}")

    # Build command with parameters using auto-detected runner path
    runner_path = _find_runner_path()
    cmd = f"{sys.executable} {runner_path} --config-name={profile} nx={nx} dt={dt} newton_v_res_tol={newton_v_res_tol}"
    subprocess.run(cmd, shell=True, check=True)

    # Load the cost and convergence status from the meta.json file
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]
        is_converged = meta.get("is_converged", False)

    return cost, is_converged


def get_energies_fem2d(profile, nx, dt, newton_v_res_tol):
    """Load energies data for a given parameter set, triggering a simulation if results are missing.

    Args:
        profile: Profile name (p1, p2, p3, etc.)
        nx: Grid resolution parameter
        dt: Time step size
        newton_v_res_tol: Newton velocity residual tolerance

    Returns:
        dict: Dictionary containing energy arrays (kin, pot, tot)
    """
    dir_path = f"sim_res/fem2d/{profile}_nx{nx}_dt{format_param_for_path(dt)}_nvrestol{format_param_for_path(newton_v_res_tol)}/"
    energies_path = os.path.join(dir_path, "energies.npz")

    # Check if energies file exists, otherwise trigger a simulation
    if not os.path.exists(energies_path):
        print(f"No energies found for parameters: nx={nx}, dt={dt}, newton_v_res_tol={newton_v_res_tol}. Triggering simulation.")
        run_sim_fem2d(profile=profile, nx=nx, dt=dt, newton_v_res_tol=newton_v_res_tol)

    # Load energies data
    energies_data = np.load(energies_path)
    return_dict = {}
    # include all keys existing in energies_data
    for key in energies_data.keys():
        return_dict[key] = energies_data[key]
    return return_dict


def compare_energies_fem2d(profile1, nx1, dt1, newton_v_res_tol1,
                           profile2, nx2, dt2, newton_v_res_tol2,
                           energy_tolerance=1e-6, var_threshold=0.01):
    """Compare energies between two fem2d simulations.

    Args:
        profile1, profile2: Profile names
        nx1, nx2: Grid resolution parameters
        dt1, dt2: Time step sizes
        newton_v_res_tol1, newton_v_res_tol2: Newton velocity residual tolerances
        energy_tolerance: Tolerance for energy comparison
        var_threshold: Threshold for energy conservation (coefficient of variation)

    Returns:
        converged (bool): True if energies are within tolerance
        metrics1 (dict): Energy metrics for case 1
        metrics2 (dict): Energy metrics for case 2
        avg_energy_diff (float): Average relative energy difference
    """
    # Load energies for both cases
    energies1 = get_energies_fem2d(profile1, nx1, dt1, newton_v_res_tol1)
    energies2 = get_energies_fem2d(profile2, nx2, dt2, newton_v_res_tol2)

    if energies1 is None or energies2 is None:
        print("Failed to load energies for one or both cases")
        return False, None, None, float('inf')

    # Check if simulation failed by checking meta.json
    dir1 = f"sim_res/fem2d/{profile1}_nx{nx1}_dt{format_param_for_path(dt1)}_nvrestol{format_param_for_path(newton_v_res_tol1)}/"
    dir2 = f"sim_res/fem2d/{profile2}_nx{nx2}_dt{format_param_for_path(dt2)}_nvrestol{format_param_for_path(newton_v_res_tol2)}/"

    meta1_path = os.path.join(dir1, "meta.json")
    meta2_path = os.path.join(dir2, "meta.json")

    # Check if either simulation failed
    if os.path.exists(meta1_path):
        with open(meta1_path, "r") as f:
            meta1 = json.load(f)
            if not meta1.get("is_converged", False):
                print(f"Simulation 1 failed: {profile1}_nx{nx1}_dt{dt1}_nvrestol{newton_v_res_tol1}")
                return False, None, None, float('inf')

    if os.path.exists(meta2_path):
        with open(meta2_path, "r") as f:
            meta2 = json.load(f)
            if not meta2.get("is_converged", False):
                print(f"Simulation 2 failed: {profile2}_nx{nx2}_dt{dt2}_nvrestol{newton_v_res_tol2}")
                return False, None, None, float('inf')

    # Compare energies - FEM2D has kinetic and elastic potential energies
    energy_types = ["kin", "pot"]
    all_relative_diffs = []

    for energy_type in energy_types:
        if energy_type in energies1 and energy_type in energies2:
            # Calculate relative difference
            energy1 = energies1[energy_type]
            energy2 = energies2[energy_type]

            # Ensure same length for comparison
            min_len = min(len(energy1), len(energy2))
            if min_len == 0:
                continue

            energy1 = energy1[:min_len]
            energy2 = energy2[:min_len]

            # Calculate relative difference using L2 norm
            eps = 1e-12
            rel_diff = np.linalg.norm(energy1 - energy2) / (np.linalg.norm(energy1) + np.linalg.norm(energy2) + eps)
            all_relative_diffs.append(rel_diff)

    # Calculate average relative difference across all energy types
    avg_energy_diff = np.mean(all_relative_diffs) if all_relative_diffs else float('inf')

    # Get case name from profile (p1, p2, p3)
    case1 = _get_case_from_profile(profile1)
    case2 = _get_case_from_profile(profile2)

    # Compute metrics for both cases
    metrics1 = compute_energy_metrics(energies1, var_threshold, case1)
    metrics2 = compute_energy_metrics(energies2, var_threshold, case2)

    # Check convergence - includes energy conservation, positivity, and energy difference
    converged = (avg_energy_diff < energy_tolerance and
                 metrics1["energy_conserved"] and metrics2["energy_conserved"] and
                 metrics1["energy_positivity_valid"] and metrics2["energy_positivity_valid"])

    print(f"Average relative energy difference: {avg_energy_diff:.2e}, variance: {metrics1['energy_variation']:.2e}, {metrics2['energy_variation']:.2e}")
    print(f"Energy tolerance: {energy_tolerance:.2e}, var_threshold: {var_threshold:.2e}")
    print(f"Energy positivity valid: {metrics1['energy_positivity_valid']}, {metrics2['energy_positivity_valid']}")
    print(f"Converged: {converged}")

    return converged, metrics1, metrics2, avg_energy_diff


def _get_case_from_profile(profile):
    """Map profile name to case name."""
    # Read the config file to get the case name
    config_path = f"run_configs/fem2d/{profile}.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            for line in f:
                if line.strip().startswith('case:'):
                    return line.split(':')[1].strip().strip('"')
    # Default fallback
    if profile == "p1":
        return "cantilever"
    elif profile == "p2":
        return "vibration_bar"
    elif profile == "p3":
        return "twisting_column"
    return "unknown"


def compute_energy_metrics(energies, var_threshold, case="cantilever"):
    """Compute energy conservation metrics for FEM2D simulations.

    Args:
        energies: Dictionary containing energy arrays
        var_threshold: Threshold for energy conservation (coefficient of variation)
        case: Case name (cantilever, vibration_bar, twisting_column)

    Returns:
        dict: Dictionary containing energy metrics
    """
    if energies is None:
        return None

    metrics = {}

    # Check energy positivity - kinetic and elastic potential energies must be ≥ 0
    positivity_violated = False
    for energy_type in ["kin", "pot"]:  # kinetic and elastic potential energies
        if energy_type in energies:
            energy_values = energies[energy_type]
            min_energy = np.min(energy_values)
            if min_energy < 0:
                positivity_violated = True
                metrics[f"{energy_type}_positivity_violated"] = True
                metrics[f"{energy_type}_min_negative"] = min_energy
            else:
                metrics[f"{energy_type}_positivity_violated"] = False
                metrics[f"{energy_type}_min_negative"] = 0.0

    metrics["energy_positivity_valid"] = not positivity_violated

    # Check if total energy is conserved (should be approximately constant)
    if "tot" in energies:
        tot_energy = energies["tot"]
        if len(tot_energy) > 1:
            # For all FEM2D cases, check all time steps
            # (no special handling needed like MPM disk_collision case)
            energy_variation = np.std(tot_energy) / (np.mean(np.abs(tot_energy)) + 1e-12)

            # Note: cantilever case has gravity, so energy is NOT conserved
            # Only vibration_bar and twisting_column should conserve energy
            if case == "cantilever":
                # For cantilever, we don't check energy conservation
                metrics["energy_conserved"] = True  # Always pass for cantilever
                metrics["energy_variation"] = energy_variation
                metrics["energy_check_period"] = "not_applicable_gravity"
            else:
                # For vibration_bar and twisting_column, check energy conservation
                metrics["energy_conserved"] = energy_variation < var_threshold
                metrics["energy_variation"] = energy_variation
                metrics["energy_check_period"] = "all_steps"
        else:
            metrics["energy_conserved"] = True
            metrics["energy_variation"] = 0.0
            metrics["energy_check_period"] = "single_step"

    # Check energy bounds
    for energy_type in ["kin", "pot", "tot"]:
        if energy_type in energies:
            energy_values = energies[energy_type]
            metrics[f"{energy_type}_min"] = np.min(energy_values)
            metrics[f"{energy_type}_max"] = np.max(energy_values)

    return metrics


def print_energy_metrics(case_name, metrics):
    """Print energy metrics in a formatted way."""
    if metrics is None:
        print(f"{case_name}: No metrics available")
        return

    print(f"\n{case_name} Energy Metrics:")
    print(f"  Energy conserved: {metrics.get('energy_conserved', 'N/A')}")
    print(f"  Energy variation: {metrics.get('energy_variation', 'N/A'):.2e}")
    print(f"  Energy check period: {metrics.get('energy_check_period', 'N/A')}")
    print(f"  Energy positivity valid: {metrics.get('energy_positivity_valid', 'N/A')}")

    # Print positivity violation details
    for energy_type in ["kin", "pot"]:
        if f"{energy_type}_positivity_violated" in metrics:
            if metrics[f"{energy_type}_positivity_violated"]:
                print(f"  {energy_type.upper()} energy positivity VIOLATED: min = {metrics.get(f'{energy_type}_min_negative', 'N/A'):.2e}")
            else:
                print(f"  {energy_type.upper()} energy positivity: OK")

    for energy_type in ["kin", "pot", "tot"]:
        if f"{energy_type}_min" in metrics:
            min_val = metrics[f"{energy_type}_min"]
            max_val = metrics[f"{energy_type}_max"]
            print(f"  {energy_type.upper()} energy: {min_val:.2e} to {max_val:.2e}")


if __name__ == "__main__":
    # Example usage
    profile = "p1"
    nx = 20
    dt = 0.0005
    newton_v_res_tol = 0.01

    cost, converged = run_sim_fem2d(profile, nx, dt, newton_v_res_tol)
    print(f"Simulation cost: {cost}, Converged: {converged}")

    energies = get_energies_fem2d(profile, nx, dt, newton_v_res_tol)
    if energies:
        print(f"Energies loaded: {list(energies.keys())}")
        print(f"Total energy shape: {energies['tot'].shape}")
