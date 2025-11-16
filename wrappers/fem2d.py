import os
import subprocess
import numpy as np
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get base directory for simulation results from environment variable
# If not set, use current directory (maintains backward compatibility)
SIM_RES_BASE_DIR = os.getenv("SIM_RES_BASE_DIR", None)
if SIM_RES_BASE_DIR:
    print(f"✅ Using custom simulation results directory: {SIM_RES_BASE_DIR}")

# Add repository root to Python path to import solvers.utils
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from solvers.utils import format_param_for_path


def _get_sim_path(relative_path):
    """Construct simulation path, using absolute path if SIM_RES_BASE_DIR is set."""
    if SIM_RES_BASE_DIR:
        return os.path.join(SIM_RES_BASE_DIR, relative_path)
    return relative_path


def _find_runner_path():
    """Automatically find the correct path to fem2d.py runner."""
    # Get current working directory
    cwd = os.getcwd()

    # List of possible runner paths relative to different working directories
    possible_paths = []

    # If working from project root (SimulCost-Bench/)
    if cwd.endswith("SimulCost-Bench"):
        possible_paths.extend(["costsci_tools/runners/fem2d.py", "runners/fem2d.py"])
    # If working from costsci_tools/ subdirectory
    elif cwd.endswith("costsci_tools") or "costsci_tools" in cwd:
        possible_paths.extend(["runners/fem2d.py", "../runners/fem2d.py", "costsci_tools/runners/fem2d.py"])

    # Add generic fallback paths
    possible_paths.extend(
        [
            "runners/fem2d.py",
            "costsci_tools/runners/fem2d.py",
            "./runners/fem2d.py",
            "../runners/fem2d.py",
            "../../runners/fem2d.py",
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
        f"Could not find fem2d.py runner in any expected location.\n"
        f"Current working directory: {cwd}\n"
        f"Searched paths: {unique_paths}\n"
        f"Please ensure the runner exists or update the search paths."
    )


def get_fem2d_data(profile, dx, cfl):
    """
    Retrieves fem2d simulation results, running the simulation if not cached.

    This function checks for both 'meta.json' (with 'cost')
    and 'energies.npz'. If any are missing or incomplete, it runs the
    simulation. It then loads and returns all data.

    Args:
        profile: Profile name (p1, p2, p3, etc.)
        dx: Grid resolution parameter
        cfl: CFL number for time step calculation

    Returns:
        tuple: (energies, cost)
            - energies (dict): Dictionary of energy arrays
            - cost (float): Simulation cost
    """
    # Create directory path based on parameters
    dir_path = _get_sim_path(f"sim_res/fem2d/{profile}_dx{dx}_cfl{format_param_for_path(cfl)}/")
    meta_path = os.path.join(dir_path, "meta.json")
    energies_path = os.path.join(dir_path, "energies.npz")

    # --- 1. Check for valid cache ---
    has_valid_cache = False
    if os.path.exists(meta_path) and os.path.exists(energies_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                # Fail fast if keys are missing
                _ = meta["cost"]
                # Note: fem2d doesn't write "is_converged" - it doesn't have convergence criteria
                has_valid_cache = True
                print(f"Using existing simulation results from {dir_path}")
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Cache at {dir_path} is corrupted ({e}). Rerunning.")

    # --- 2. Run simulation if cache is missing or invalid ---
    if not has_valid_cache:
        print(f"Running new simulation: dx={dx}, cfl={cfl}")

        runner_path = _find_runner_path()
        if SIM_RES_BASE_DIR:
            dump_dir = os.path.join(SIM_RES_BASE_DIR, f"sim_res/fem2d/{profile}")
            cmd = f"{sys.executable} {runner_path} --config-name={profile} dx={dx} cfl={cfl} dump_dir={dump_dir}"
        else:
            cmd = f"{sys.executable} {runner_path} --config-name={profile} dx={dx} cfl={cfl}"

        # This will raise CalledProcessError if the command fails
        subprocess.run(cmd, shell=True, check=True)

    # --- 3. Load results (files *must* exist now) ---

    # Load metadata - Will raise FileNotFoundError, JSONDecodeError, or KeyError
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]
        # is_converged = meta["is_converged"]

    # Load energies - Will raise FileNotFoundError
    energies_data = np.load(energies_path)
    energies = {}
    for key in energies_data.keys():
        energies[key] = energies_data[key]

    return energies, cost


def compare_energies_fem2d(profile1, dx1, cfl1, profile2, dx2, cfl2, energy_tolerance, var_threshold):
    """Compare energies between two fem2d simulations.

    Args:
        profile1, profile2: Profile names
        dx1, dx2: Grid resolution parameters
        cfl1, cfl2: CFL numbers for time step calculation
        energy_tolerance: Tolerance for energy comparison
        var_threshold: Threshold for energy conservation (coefficient of variation)

    Returns:
        converged (bool): True if energies are within tolerance
        metrics1 (dict): Energy metrics for case 1
        metrics2 (dict): Energy metrics for case 2
        avg_energy_diff (float): Average relative energy difference
    """
    # Load energies and metadata for both cases
    energies1, _ = get_fem2d_data(profile1, dx1, cfl1)
    energies2, _ = get_fem2d_data(profile2, dx2, cfl2)

    # # Check if either simulation failed
    # if not is_converged1:
    #     print(f"Simulation 1 failed: {profile1}_nx{nx1}_dt{dt1}_nvrestol{newton_v_res_tol1}")
    #     return False, None, None, float("inf")
    # if not is_converged2:
    #     print(f"Simulation 2 failed: {profile2}_nx{nx2}_dt{dt2}_nvrestol{newton_v_res_tol2}")
    #     return False, None, None, float("inf")

    # Compare energies - FEM2D has kinetic, elastic potential, and gravitational energies
    energy_types = ["kin", "pot", "gra"]
    all_relative_diffs = []

    for energy_type in energy_types:
        energy1 = energies1[energy_type]
        energy2 = energies2[energy_type]

        # Ensure same length for comparison
        if len(energy1) != len(energy2):
            raise ValueError(
                f"Energy arrays for '{energy_type}' have mismatched lengths: {len(energy1)} vs {len(energy2)}"
            )

        if len(energy1) == 0:
            raise ValueError(f"Energy array for '{energy_type}' is empty.")

        # Calculate relative difference using L2 norm
        eps = 1e-12
        rel_diff = np.linalg.norm(energy1 - energy2) / (np.linalg.norm(energy1) + np.linalg.norm(energy2) + eps)
        # print("diff", np.linalg.norm(energy1 - energy2))
        # print("norm", (np.linalg.norm(energy1) + np.linalg.norm(energy2) + eps))
        # exit(1)
        all_relative_diffs.append(rel_diff)

    # Calculate average relative difference across all energy types
    avg_energy_diff = np.mean(all_relative_diffs)

    # Compute metrics for both cases
    metrics1 = compute_energy_metrics(energies1, var_threshold)
    metrics2 = compute_energy_metrics(energies2, var_threshold)

    # Check convergence - includes energy conservation, positivity, and energy difference
    converged = (
        avg_energy_diff < energy_tolerance
        and metrics1["energy_conserved"]
        and metrics2["energy_conserved"]
        and metrics1["energy_positivity_valid"]
        and metrics2["energy_positivity_valid"]
    )

    print(
        f"Average relative energy difference: {avg_energy_diff:.2e}, variance: {metrics1['energy_variation']:.2e}, {metrics2['energy_variation']:.2e}"
    )
    print(f"Energy tolerance: {energy_tolerance:.2e}, var_threshold: {var_threshold:.2e}")
    print(f"Energy positivity valid: {metrics1['energy_positivity_valid']}, {metrics2['energy_positivity_valid']}")
    print(f"Converged: {converged}")

    return converged, metrics1, metrics2, avg_energy_diff


def compute_energy_metrics(energies, var_threshold):
    """Compute energy conservation metrics for FEM2D simulations.

    Args:
        energies: Dictionary containing energy arrays
        var_threshold: Threshold for energy conservation (coefficient of variation)

    Returns:
        dict: Dictionary containing energy metrics
    """
    metrics = {}

    # Check energy positivity - kinetic and elastic potential energies must be >= 0
    # Note: gravitational potential energy (gra) is NOT checked because it depends on reference height
    positivity_violated = False
    for energy_type in ["kin", "pot"]:  # kinetic and elastic potential energies only
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
    tot_energy = energies["tot"]
    if len(tot_energy) > 1:
        # Calculate scale based on max peak-to-peak of component energies
        kin_range = np.ptp(energies["kin"])  # Peak-to-Peak (max - min)
        pot_range = np.ptp(energies["pot"])
        gra_range = np.ptp(energies["gra"])

        scale = np.max([kin_range, pot_range, gra_range]) + 1e-12

        energy_variation = np.std(tot_energy) / scale

        # For all cases, check energy conservation
        metrics["energy_conserved"] = energy_variation < var_threshold
        metrics["energy_variation"] = energy_variation
        metrics["energy_check_period"] = "all_steps"
    else:
        # Fail fast if simulation is too short
        raise ValueError("Cannot compute energy metrics: simulation has one or zero time steps.")

    # Check energy bounds
    for energy_type in ["kin", "pot", "gra", "tot"]:
        energy_values = energies[energy_type]
        metrics[f"{energy_type}_min"] = np.min(energy_values)
        metrics[f"{energy_type}_max"] = np.max(energy_values)

    return metrics


if __name__ == "__main__":
    ps = ["p1", "p2", "p3"]
    dx_values = {
        "p1": (0.25, 0.125),
        "p2": (0.625, 0.3125),
        "p3": (0.05, 0.025),
    }
    for p in ps:
        dx1, dx2 = dx_values[p]
        print(compare_energies_fem2d(p, dx1, 0.5, p, dx2, 0.5, 0.01, 0.01))
