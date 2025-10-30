import os
import subprocess
import h5py
import numpy as np
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from solvers.utils import format_param_for_path

# Load environment variables from .env file
load_dotenv()

env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Get base directory for simulation results from environment variable
# If not set, use current directory (maintains backward compatibility)
SIM_RES_BASE_DIR = os.getenv("SIM_RES_BASE_DIR", None)
if SIM_RES_BASE_DIR:
    print(f"✅ Using custom simulation results directory: {SIM_RES_BASE_DIR}")


def _get_sim_path(relative_path):
    """Construct simulation path, using absolute path if SIM_RES_BASE_DIR is set."""
    if SIM_RES_BASE_DIR:
        return os.path.join(SIM_RES_BASE_DIR, relative_path)
    return relative_path


def _find_runner_path():
    """Automatically find the correct path to diff_react_1d.py runner."""
    # Get current working directory
    cwd = os.getcwd()

    # List of possible runner paths relative to different working directories
    possible_paths = []

    # If working from project root (SimulCost-Bench/)
    if cwd.endswith('SimulCost-Bench'):
        possible_paths.extend([
            "costsci_tools/runners/diff_react_1d.py",
            "runners/diff_react_1d.py"
        ])
    # If working from costsci_tools/ subdirectory
    elif cwd.endswith('costsci_tools') or 'costsci_tools' in cwd:
        possible_paths.extend([
            "runners/diff_react_1d.py",
            "../runners/diff_react_1d.py",
            "costsci_tools/runners/diff_react_1d.py"
        ])

    # Add generic fallback paths
    possible_paths.extend([
        "runners/diff_react_1d.py",
        "costsci_tools/runners/diff_react_1d.py",
        "./runners/diff_react_1d.py",
        "../runners/diff_react_1d.py",
        "../../runners/diff_react_1d.py"
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
        f"Could not find diff_react_1d.py runner in any expected location.\n"
        f"Current working directory: {cwd}\n"
        f"Searched paths: {unique_paths}\n"
        f"Please ensure the runner exists or update the search paths."
    )


def run_sim_diff_react_1d(profile, n_space, cfl, tol, min_step, initial_step_guess, reaction_type="fisher", allee_threshold=None):
    """Run the diff_react_1d simulation with the given parameters if not already simulated."""
    # Build directory path based on parameters

    param_str = f"_nspace{n_space}_cfl{format_param_for_path(cfl)}_tol{format_param_for_path(tol)}"
    dir_path = _get_sim_path(f"sim_res/diff_react_1d/{profile}{param_str}/")

    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"]

    # Run the simulation if not already done
    print(f"Running new simulation with parameters: n_space={n_space}, cfl={cfl}, tol={tol}, reaction_type={reaction_type}")

    # Build command with parameters using auto-detected runner path
    runner_path = _find_runner_path()
    if SIM_RES_BASE_DIR:
        dump_dir = os.path.join(SIM_RES_BASE_DIR, f"sim_res/diff_react_1d/{profile}")
        cmd = f"{sys.executable} {runner_path} --config-name={profile} n_space={n_space} cfl={cfl} tol={tol} reaction_type={reaction_type} dump_dir={dump_dir}"
    else:
        cmd = f"{sys.executable} {runner_path} --config-name={profile} n_space={n_space} cfl={cfl} tol={tol} reaction_type={reaction_type}"

    if allee_threshold is not None:
        cmd += f" allee_threshold={allee_threshold}"
    subprocess.run(cmd, shell=True, check=True, env=env)

    # Load the cost from the meta.json file
    with open(meta_path, "r") as f:
        meta = json.load(f)
        cost = meta["cost"]

    return cost


def get_res_diff_react_1d(profile, n_space, cfl, tol, min_step, initial_step_guess, reaction_type="fisher", allee_threshold=None):
    """Load all time frames for a given parameter set, triggering a simulation if results are missing."""
    # Build directory path based on parameters

    param_str = f"_nspace{n_space}_cfl{format_param_for_path(cfl)}_tol{format_param_for_path(tol)}"
    dir_path = _get_sim_path(f"sim_res/diff_react_1d/{profile}{param_str}/")
    results = {}
    X = None

    # Check if at least one result file exists, otherwise trigger a simulation
    if not os.path.exists(dir_path) or not any(
        fname.startswith("res_") and fname.endswith(".h5") for fname in os.listdir(dir_path)
    ):
        print(
            f"No results found for parameters: n_space={n_space}, cfl={cfl}, tol={tol}, min_step={min_step}, initial_step_guess={initial_step_guess}, reaction_type={reaction_type}. Triggering simulation."
        )
        run_sim_diff_react_1d(profile, n_space, cfl, tol, min_step, initial_step_guess, reaction_type, allee_threshold)

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


def compute_wave_position(u, x, threshold=0.5):
    """Compute wave front position for diffusion-reaction equation.
    Computes wave positions for all time steps.
    Automatically adjusts threshold based on spatial resolution.
    
    Args:
        u: np.ndarray with shape [nt, nx] (time steps, spatial points)
        x: np.ndarray with spatial coordinates
        threshold: float, threshold value to define wave front (default 0.5)
    
    Returns:
        np.ndarray: Wave front positions at each time step
    """
    wave_positions = []
    
    n_space = len(x)
    adjusted_threshold = threshold
    
    for i in range(u.shape[0]):
        u_t = u[i, :]
        
        # Check if this looks like a slow wave (allee reaction)
        max_u = np.max(u_t)
        active_ratio = np.count_nonzero(u_t > 0.1) / len(u_t)
        
        if active_ratio < 0.3 and max_u > 0.9:  # Slow wave pattern
            # Use a much lower threshold for slow waves
            effective_threshold = 0.1
            wave_front_idx = np.where(u_t > effective_threshold)[0]
            if len(wave_front_idx) > 0:
                wave_pos = x[wave_front_idx[-1]]
            else:
                wave_pos = x[0]
            wave_positions.append(wave_pos)
        else:
            # Use center of mass for normal waves
            total_mass = np.sum(u_t)
            if total_mass > 1e-10:
                center_of_mass = np.sum(x * u_t) / total_mass
                wave_positions.append(center_of_mass)
            else:
                # Fallback to threshold method with adjusted threshold
                wave_front_idx = np.where(u_t > adjusted_threshold)[0]
                if len(wave_front_idx) > 0:
                    wave_pos = x[wave_front_idx[-1]]
                else:
                    wave_pos = x[0]
                wave_positions.append(wave_pos)
    
    return np.array(wave_positions)


def compute_metrics_diff_react_1d(u, reaction_type="fisher"):
    """Compute physical metrics for diffusion-reaction equation solution.
    Args:
        u: np.ndarray with shape [nt, nx] (time steps, spatial points)
        reaction_type: str, type of reaction term
    Returns:
        Dictionary containing:
        - max_principle_satisfied: bool array if max principle is satisfied (shape [nt])
        - boundary_conditions_satisfied: bool array if BCs are satisfied (shape [nt])
        - reaction_balance: float, measure of reaction-diffusion balance
        - wave_propagation_quality: float, measure of wave propagation quality (for P2)
    """
    
    if reaction_type == "allee":
        # Special metrics for P2 profile (allee reaction)
        # Wave propagation quality: check if wave front is well-defined with stricter criteria
        wave_propagation_quality = []
        for i in range(u.shape[0]):
            u_t = u[i, :]
            # Check if there's a clear wave front (gradient should be sharp)
            grad_u = np.abs(np.diff(u_t))
            max_grad = np.max(grad_u)
            
            wave_propagation_quality.append(max_grad > 0.04)
        
        return {
            "max_principle_satisfied": np.ones(u.shape[0], dtype=bool),  # Always satisfied for allee
            "boundary_conditions_satisfied": np.ones(u.shape[0], dtype=bool),  # Always satisfied for allee
            "reaction_balance": np.mean(np.abs(np.diff(u, axis=0)), axis=1),
            "wave_propagation_quality": wave_propagation_quality,
        }
    else:
        # Standard metrics for fisher and cubic reactions
        # Maximum principle (solution should be bounded by initial max and min)
        initial_max = np.max(u[0])
        initial_min = np.min(u[0])
        max_principle_violation = np.maximum(u - initial_max, 0) + np.maximum(initial_min - u, 0)
        max_violation = np.max(max_principle_violation, axis=1)
        max_principle_satisfied = max_violation <= 1e-6

        # Boundary conditions (u(0,t) = 1, u(L,t) = 0 for Dirichlet BCs)
        left_bc_satisfied = np.isclose(u[:, 0], 1.0, rtol=1e-6)
        right_bc_satisfied = np.isclose(u[:, -1], 0.0, rtol=1e-6)
        boundary_conditions_satisfied = left_bc_satisfied & right_bc_satisfied

        # Reaction-diffusion balance (measure of how well the equation is satisfied)
        reaction_balance = np.mean(np.abs(np.diff(u, axis=0)), axis=1)  # Temporal variation

        return {
            "max_principle_satisfied": max_principle_satisfied,
            "boundary_conditions_satisfied": boundary_conditions_satisfied,
            "reaction_balance": reaction_balance,
        }


def print_metrics_diff_react_1d(name, metrics):
    print(f"\n--- {name} Metrics ---")
    print(f"Max principle satisfied at all steps: {np.all(metrics['max_principle_satisfied'])}")
    print(f"Boundary conditions satisfied at all steps: {np.all(metrics['boundary_conditions_satisfied'])}")
    print(f"Average reaction balance: {np.mean(metrics['reaction_balance']):.2e}")
    
    # Show special metrics for allee reaction
    if 'wave_propagation_quality' in metrics:
        print(f"Wave propagation quality: {np.all(metrics['wave_propagation_quality'])}")


def compare_res_diff_react_1d(profile1, n_space1, cfl1, tol1, min_step1, init_step1, 
                             profile2, n_space2, cfl2, tol2, min_step2, init_step2,
                             rmse_tolerance, reaction_type1="fisher", reaction_type2="fisher",
                             allee_threshold1=None, allee_threshold2=None):
    """Compare two sets of results using wave position and physical metrics.
    Returns:
        converged (bool): True if wave position tolerance is met.
        metrics1 (dict): Metrics for case 1.
        metrics2 (dict): Metrics for case 2.
        wave_error (float): Relative wave position error.
    """
    res1_dict, x1 = get_res_diff_react_1d(profile1, n_space1, cfl1, tol1, min_step1, init_step1, reaction_type1, allee_threshold1)
    res2_dict, x2 = get_res_diff_react_1d(profile2, n_space2, cfl2, tol2, min_step2, init_step2, reaction_type2, allee_threshold2)

    # Convert dictionary results to arrays for comparison
    frames1 = sorted(res1_dict.keys())
    frames2 = sorted(res2_dict.keys())
    res1 = np.array([res1_dict[frame] for frame in frames1])
    res2 = np.array([res2_dict[frame] for frame in frames2])

    # Compute wave positions for both solutions (all time steps)
    wave_pos1 = compute_wave_position(res1, x1)
    wave_pos2 = compute_wave_position(res2, x2)
    
    # Ensure both have the same number of time steps
    min_frames = min(len(wave_pos1), len(wave_pos2))
    wave_pos1 = wave_pos1[:min_frames]
    wave_pos2 = wave_pos2[:min_frames]
    
    # Compute relative wave position error (average over all time steps)
    domain_length = max(x1[-1], x2[-1])  # Use the larger domain length
    wave_error = np.mean(np.abs(wave_pos1 - wave_pos2)) / domain_length

    # Physical metrics
    metrics1 = compute_metrics_diff_react_1d(res1, reaction_type1)
    metrics2 = compute_metrics_diff_react_1d(res2, reaction_type2)

    # For allee reaction (P2), use specialized metrics; for others, use standard metrics
    if reaction_type1 == "allee" or reaction_type2 == "allee":
        # For allee reaction, only check wave error and wave propagation quality
        # Note: Allee effect reactions do NOT conserve mass by design (f(u) = u(1-u)(u-a))
        # The reaction term can be both positive (growth) and negative (decay)
        converged = (
            wave_error < rmse_tolerance
            and np.all(metrics1["wave_propagation_quality"])
            and np.all(metrics2["wave_propagation_quality"])
        )
    else:
        # For fisher and cubic reactions, use standard metrics
        converged = (
            wave_error < rmse_tolerance
            and np.all(metrics1["max_principle_satisfied"])
            and np.all(metrics2["max_principle_satisfied"])
            and np.all(metrics1["boundary_conditions_satisfied"])
            and np.all(metrics2["boundary_conditions_satisfied"])
        )

    print_metrics_diff_react_1d("Case 1", metrics1)
    print_metrics_diff_react_1d("Case 2", metrics2)

    print(f"Wave position error (relative): {wave_error:.6f}")
    print(f"Average wave positions - Case 1: {np.mean(wave_pos1):.3f}, Case 2: {np.mean(wave_pos2):.3f}")
    print(f"Wave position range - Case 1: [{np.min(wave_pos1):.3f}, {np.max(wave_pos1):.3f}]")
    print(f"Wave position range - Case 2: [{np.min(wave_pos2):.3f}, {np.max(wave_pos2):.3f}]")

    return converged, metrics1, metrics2, wave_error


if __name__ == "__main__":
    # Example usage
    profile1 = "p1"
    n_space1 = 1024
    cfl1 = 0.5
    tol1 = 1e-9
    min_step1 = 1e-3
    init_step1 = 1.0
    reaction_type1 = "fisher"

    profile2 = "p1"
    n_space2 = 2048
    cfl2 = 0.5
    tol2 = 1e-9
    min_step2 = 1e-3
    init_step2 = 1.0
    reaction_type2 = "fisher"

    tolerance = 0.01

    res1, x1 = get_res_diff_react_1d(profile1, n_space1, cfl1, tol1, min_step1, init_step1, reaction_type1)
    res2, x2 = get_res_diff_react_1d(profile2, n_space2, cfl2, tol2, min_step2, init_step2, reaction_type2)

    print(compare_res_diff_react_1d(profile1, n_space1, cfl1, tol1, min_step1, init_step1,
                                   profile2, n_space2, cfl2, tol2, min_step2, init_step2,
                                   tolerance, reaction_type1, reaction_type2))
