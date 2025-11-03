import os
import subprocess
import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
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


def _get_sim_path(relative_path):
    """Construct simulation path, using absolute path if SIM_RES_BASE_DIR is set."""
    if SIM_RES_BASE_DIR:
        return os.path.join(SIM_RES_BASE_DIR, relative_path)
    return relative_path


def _find_runner_path():
    """Automatically find the correct path to hasegawa_mima_linear.py runner."""
    # Get current working directory
    cwd = os.getcwd()

    # List of possible runner paths relative to different working directories
    possible_paths = []

    # If working from project root (SimulCost-Bench/)
    if cwd.endswith('SimulCost-Bench'):
        possible_paths.extend([
            "costsci_tools/runners/hasegawa_mima_linear.py",
            "runners/hasegawa_mima_linear.py"
        ])
    # If working from costsci_tools/ subdirectory
    elif cwd.endswith('costsci_tools') or 'costsci_tools' in cwd:
        possible_paths.extend([
            "runners/hasegawa_mima_linear.py",
            "../runners/hasegawa_mima_linear.py",
            "costsci_tools/runners/hasegawa_mima_linear.py"
        ])

    # Add generic fallback paths
    possible_paths.extend([
        "runners/hasegawa_mima_linear.py",
        "costsci_tools/runners/hasegawa_mima_linear.py",
        "./runners/hasegawa_mima_linear.py",
        "../runners/hasegawa_mima_linear.py",
        "../../runners/hasegawa_mima_linear.py"
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
        f"Could not find hasegawa_mima_linear.py runner in any expected location.\n"
        f"Current working directory: {cwd}\n"
        f"Searched paths: {unique_paths}\n"
        f"Please ensure the runner exists or update the search paths."
    )


def run_sim_hasegawa_mima_linear(profile, N, dt, cg_atol, analytical):
    """Run the Hasegawa-Mima linear simulation with the given parameters if not already simulated."""
    if analytical:
        method_suffix = "_analytical"
        dir_path = _get_sim_path(f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}" + method_suffix + "/")
    else:
        method_suffix = "_numerical"
        dir_path = _get_sim_path(
            f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}_cg_{cg_atol:.2e}" + method_suffix + "/"
        )
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"]

    # Run the simulation if not already done
    method_name = "analytical" if analytical else "numerical"
    print(f"Running new {method_name} simulation with parameters: N={N}, dt={dt}, cg_atol={cg_atol:.2e}")
    runner_path = _find_runner_path()
    if SIM_RES_BASE_DIR:
        dump_dir = os.path.join(SIM_RES_BASE_DIR, f"sim_res/hasegawa_mima_linear/{profile}")
        cmd = f"{sys.executable} {runner_path} --config-name={profile} N={N} dt={dt} cg_atol={cg_atol} analytical={analytical} dump_dir={dump_dir}"
    else:
        cmd = f"{sys.executable} {runner_path} --config-name={profile} N={N} dt={dt} cg_atol={cg_atol} analytical={analytical}"
    subprocess.run(cmd, shell=True, check=True)

    # Read the meta.json to get cost
    with open(meta_path, "r") as f:
        meta = json.load(f)
        return meta["cost"]


def run_simulation(config_path, N, dt, cg_atol, analytical, verbose, **kwargs):
    """
    Run Hasegawa-Mima linear simulation with given parameters.

    Args:
        config_path: Path to YAML config file (optional)
        N: Grid resolution (main tunable parameter)
        dt: Time step (main tunable parameter)
        cg_atol: CG solver tolerance (tunable parameter)
        analytical: Use analytical solution (True) or numerical (False)
        verbose: Enable verbose output
        **kwargs: Additional parameters to override

    Returns:
        dict: Simulation results including error metric
    """
    runner_path = _find_runner_path()

    if config_path:
        cmd = [sys.executable, runner_path, f"--config-path={config_path}"]
    else:
        cmd = [sys.executable, runner_path]

    # Override parameters
    cmd.extend([f"N={N}", f"dt={dt}", f"cg_atol={cg_atol}", f"analytical={analytical}", f"verbose={verbose}"])

    for key, value in kwargs.items():
        cmd.append(f"{key}={value}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print("STDOUT:", result.stdout)
        return {"success": True, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": str(e), "stdout": e.stdout, "stderr": e.stderr}


def load_results(sim_dir, frame):
    """
    Load simulation results from output directory.

    Args:
        sim_dir: Simulation output directory
        frame: Frame number to load (default: 0)

    Returns:
        dict: Loaded simulation results
    """
    h5_file = os.path.join(sim_dir, f"frame_{frame:04d}.h5")
    json_file = os.path.join(sim_dir, f"frame_{frame:04d}.json")

    results = {}

    if os.path.exists(h5_file):
        with h5py.File(h5_file, "r") as f:
            results["phi"] = f["phi"][:]
            results["coordinates_x"] = f["coordinates_x"][:]
            results["coordinates_y"] = f["coordinates_y"][:]
            results["time"] = f.attrs["time"]
            results["N"] = f.attrs["N"]
            results["dt"] = f.attrs["dt"]

    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            json_data = json.load(f)
            results.update(json_data)

    return results


def get_error_metric(numerical_sim_dir):
    """
    Extract error metric by comparing numerical solution with analytical solution.

    Args:
        numerical_sim_dir: Numerical simulation output directory

    Returns:
        float: Mean L2 error compared to analytical solution, or None if comparison fails
    """
    # Apply path transformation to support SIM_RES_BASE_DIR
    numerical_sim_dir = _get_sim_path(numerical_sim_dir)

    # Load metadata to get analytical reference directory
    meta_file = os.path.join(numerical_sim_dir, "meta.json")
    if not os.path.exists(meta_file):
        print(f"Warning: meta.json not found in {numerical_sim_dir}")
        return None

    with open(meta_file, "r") as f:
        meta = json.load(f)

    # Check if this is an analytical run (no error needed)
    if meta.get("analytical", False):
        return 0.0

    # Get analytical reference directory
    analytical_sim_dir = meta.get("analytical_reference_dir", None)
    if analytical_sim_dir is None:
        print(f"Warning: analytical_reference_dir not found in metadata")
        return None

    if not os.path.exists(analytical_sim_dir):
        print(f"Warning: Analytical reference directory not found: {analytical_sim_dir}")
        return None

    # Compare numerical and analytical solutions
    comparison_result = compare_with_analytical(numerical_sim_dir, analytical_sim_dir)

    if comparison_result.get("success", False):
        return comparison_result.get("mean_l2_error", None)
    else:
        print(f"Warning: Comparison failed - {comparison_result.get('reason', 'Unknown reason')}")
        return None


def compare_with_analytical(numerical_sim_dir, analytical_sim_dir):
    """
    Compare numerical solution with analytical solution.

    Args:
        numerical_sim_dir: Directory with numerical results
        analytical_sim_dir: Directory with analytical results
        save_path: Path to save comparison plot

    Returns:
        dict: Comparison results including error metrics
    """
    # Load both solutions
    numerical_results = []
    analytical_results = []

    # Find all frame files
    frame_files = sorted([f for f in os.listdir(numerical_sim_dir) if f.startswith("frame_") and f.endswith(".h5")])

    for frame_file in frame_files:
        frame_num = int(frame_file.split("_")[1].split(".")[0])

        num_result = load_results(numerical_sim_dir, frame_num)
        ana_result = load_results(analytical_sim_dir, frame_num)

        if num_result and ana_result:
            numerical_results.append(num_result)
            analytical_results.append(ana_result)

    if not numerical_results:
        return {"success": False, "reason": "No valid results found"}

    # Calculate error metrics (L2 norm only)
    l2_errors = []
    times = []

    for num_res, ana_res in zip(numerical_results, analytical_results):
        phi_num = np.array(num_res["phi"])
        phi_ana = np.array(ana_res["phi"])
        diff = phi_num - phi_ana

        l2_error = np.sqrt(np.mean(diff**2))

        l2_errors.append(l2_error)
        times.append(num_res["time"])

    # Note: Plotting removed from wrapper - visualization happens in solver's dump() method

    return {
        "success": True,
        "times": times,
        "l2_errors": l2_errors,
        "mean_l2_error": np.mean(l2_errors),
        "max_l2_error": np.max(l2_errors),
    }
