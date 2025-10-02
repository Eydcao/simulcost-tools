import os
import subprocess
import numpy as np
import json
import sys
from pathlib import Path

# Fixed radii value for all simulations (not a tunable parameter)
FIXED_RADII = 1.5

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


def run_sim_unstruct_mpm(profile, nx, n_part, cfl, case="cantilever"):
    """Run the unstruct_mpm simulation with the given parameters if not already simulated.
    Note: radii is fixed at FIXED_RADII for all simulations.
    """
    # Create directory path based on parameters (matching solver format)
    dir_path = f"sim_res/unstruct_mpm/{profile}_nx{format_param_for_path(nx)}_npart{n_part}_cfl{format_param_for_path(cfl)}_radii{format_param_for_path(FIXED_RADII)}/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta and "is_converged" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"], meta["is_converged"]

    # Run the simulation if not already done
    print(f"Running new simulation with parameters: nx={nx}, n_part={n_part}, cfl={cfl}, case={case}")
    cmd = f"{sys.executable} runners/unstruct_mpm.py --config-name={profile} nx={nx} n_part={n_part} cfl={cfl} case={case}"
    subprocess.run(cmd, shell=True, check=True)

    # Load the cost and convergence status from the meta.json file
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]
        is_converged = meta.get("is_converged", False)

    return cost, is_converged


def get_energies_unstruct_mpm(profile, nx, n_part, cfl, case="cantilever"):
    """Load energies data for a given parameter set, triggering a simulation if results are missing.
    Note: radii is fixed at FIXED_RADII for all simulations.
    """
    dir_path = f"sim_res/unstruct_mpm/{profile}_nx{format_param_for_path(nx)}_npart{n_part}_cfl{format_param_for_path(cfl)}_radii{format_param_for_path(FIXED_RADII)}/"
    energies_path = os.path.join(dir_path, "energies.npz")

    # Check if energies file exists, otherwise trigger a simulation
    if not os.path.exists(energies_path):
        print(f"No energies found for parameters: nx={nx}, n_part={n_part}, cfl={cfl}. Triggering simulation.")
        run_sim_unstruct_mpm(profile=profile, nx=nx, n_part=n_part, cfl=cfl, case=case)

    # Load energies data
    energies_data = np.load(energies_path)
    return_dict ={}
    # include all keys existing in energies_data
    for key in energies_data.keys():
        return_dict[key] = energies_data[key]
    return return_dict


def compare_energies_unstruct_mpm(profile1, nx1, n_part1, cfl1,
                                  profile2, nx2, n_part2, cfl2,
                                  case1="cantilever", case2="cantilever",
                                  energy_tolerance=1e-6, var_threshold=0.01):
    """Compare energies between two unstruct_mpm simulations.
    Note: radii is fixed at FIXED_RADII for all simulations.

    Args:
        profile1, profile2: Profile names
        nx1, nx2: Grid resolution parameters
        n_part1, n_part2: Number of particles per cell
        cfl1, cfl2: CFL numbers
        case1, case2: Simulation cases
        energy_tolerance: Tolerance for energy comparison

    Returns:
        converged (bool): True if energies are within tolerance
        metrics1 (dict): Energy metrics for case 1
        metrics2 (dict): Energy metrics for case 2
        avg_energy_diff (float): Average relative energy difference
    """
    # Load energies for both cases
    energies1 = get_energies_unstruct_mpm(profile1, nx1, n_part1, cfl1, case1)
    energies2 = get_energies_unstruct_mpm(profile2, nx2, n_part2, cfl2, case2)

    if energies1 is None or energies2 is None:
        print("Failed to load energies for one or both cases")
        return False, None, None, float('inf')

    # Check if simulation failed by checking meta.json
    dir1 = f"sim_res/unstruct_mpm/{profile1}_nx{format_param_for_path(nx1)}_npart{n_part1}_cfl{format_param_for_path(cfl1)}_radii{format_param_for_path(FIXED_RADII)}/"
    dir2 = f"sim_res/unstruct_mpm/{profile2}_nx{format_param_for_path(nx2)}_npart{n_part2}_cfl{format_param_for_path(cfl2)}_radii{format_param_for_path(FIXED_RADII)}/"
    
    meta1_path = os.path.join(dir1, "meta.json")
    meta2_path = os.path.join(dir2, "meta.json")
    
    # Check if either simulation failed
    if os.path.exists(meta1_path):
        with open(meta1_path, "r") as f:
            meta1 = json.load(f)
            if not meta1.get("is_converged", False):
                print(f"Simulation 1 failed: {profile1}_nx{nx1}_npart{n_part1}_cfl{cfl1}")
                return False, None, None, float('inf')

    if os.path.exists(meta2_path):
        with open(meta2_path, "r") as f:
            meta2 = json.load(f)
            if not meta2.get("is_converged", False):
                print(f"Simulation 2 failed: {profile2}_nx{nx2}_npart{n_part2}_cfl{cfl2}")
                return False, None, None, float('inf')
    
    # Compare energies
    energy_types = ["pot", "kin", "gra", "tot"]
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
            
            # Calculate relative difference
            eps = 1e-12
            # use l2 relative difference
            rel_diff = np.linalg.norm(energy1 - energy2) / (np.linalg.norm(energy1) + np.linalg.norm(energy2) + eps)
            all_relative_diffs.append(rel_diff)
    
    # Calculate average relative difference across all energy types and time steps
    avg_energy_diff = np.mean(all_relative_diffs) if all_relative_diffs else float('inf')
    
    # Compute metrics for both cases
    metrics1 = compute_energy_metrics(energies1, var_threshold)
    metrics2 = compute_energy_metrics(energies2, var_threshold)
    
    # Check convergence
    converged = avg_energy_diff < energy_tolerance and metrics1["energy_conserved"] and metrics2["energy_conserved"]
    
    print(f"Average relative energy difference: {avg_energy_diff:.2e}, variance: {metrics1['energy_variation']:.2e}")
    print(f"Energy tolerance: {energy_tolerance:.2e}, var_threshold: {var_threshold:.2e}")
    print(f"Converged: {converged}")
    
    return converged, metrics1, metrics2, avg_energy_diff


def compute_energy_metrics(energies, var_threshold):
    """Compute energy conservation metrics."""
    if energies is None:
        return None
        
    metrics = {}
    
    # Check if total energy is conserved (should be approximately constant)
    if "tot" in energies:
        tot_energy = energies["tot"]
        if len(tot_energy) > 1:
            # Energy conservation: total energy should be relatively constant
            energy_variation = np.std(tot_energy) / (np.mean(np.abs(tot_energy)) + 1e-12)
            metrics["energy_conserved"] = energy_variation < var_threshold  # 1% variation threshold
            metrics["energy_variation"] = energy_variation
        else:
            metrics["energy_conserved"] = True
            metrics["energy_variation"] = 0.0
    
    # Check if energies are positive (physical constraint)
    for energy_type in ["pot", "kin", "gra", "tot"]:
        if energy_type in energies:
            energy_values = energies[energy_type]
            metrics[f"{energy_type}_positive"] = np.all(energy_values >= 0)
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
    
    for energy_type in ["pot", "kin", "gra", "tot"]:
        if f"{energy_type}_positive" in metrics:
            print(f"  {energy_type.upper()} energy: {metrics[f'{energy_type}_min']:.2e} to {metrics[f'{energy_type}_max']:.2e} (positive: {metrics[f'{energy_type}_positive']})")


if __name__ == "__main__":
    # Example usage
    profile = "p1"
    nx = 20
    n_part = 2
    cfl = 0.001

    cost, converged = run_sim_unstruct_mpm(profile, nx, n_part, cfl)
    print(f"Simulation cost: {cost}, Converged: {converged}")

    energies = get_energies_unstruct_mpm(profile, nx, n_part, cfl)
    if energies:
        print(f"Energies loaded: {list(energies.keys())}")
        print(f"Total energy shape: {energies['tot'].shape}")
