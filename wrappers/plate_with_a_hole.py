import os
import subprocess
import h5py
import numpy as np
import json
import matplotlib.pyplot as plt


def _find_runner_path():
    """Automatically find the correct path to plate_with_a_hole.py runner."""
    cwd = os.getcwd()

    possible_paths = []

    if cwd.endswith('SimulCost-Bench'):
        possible_paths.extend([
            "costsci_tools/runners/plate_with_a_hole.py",
            "runners/plate_with_a_hole.py"
        ])
    elif cwd.endswith('costsci_tools') or 'costsci_tools' in cwd:
        possible_paths.extend([
            "runners/plate_with_a_hole.py",
            "../runners/plate_with_a_hole.py",
            "costsci_tools/runners/plate_with_a_hole.py"
        ])
    else:
        possible_paths.extend([
            "runners/plate_with_a_hole.py",
            "costsci_tools/runners/plate_with_a_hole.py",
            "./runners/plate_with_a_hole.py"
        ])

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Could not find plate_with_a_hole.py runner. Searched: {possible_paths}")


def run_sim_plate_with_a_hole(profile, nx, ny):
    """Run the plate_with_a_hole simulation with the given parameters if not already simulated."""
    dir_path = f"sim_res/plate_with_a_hole/{profile}_nx_{nx}_ny_{ny}/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"]

    # Run the simulation if not already done
    print(f"Running new simulation with parameters: nx={nx}, ny={ny}")
    runner_path = _find_runner_path()
    cmd = f"PYTHONPATH=/home/yadi/costsci-tools python {runner_path} --config-name={profile} nx={nx} ny={ny}"
    subprocess.run(cmd, shell=True, check=True)

    # Read the meta.json to get cost
    with open(meta_path, "r") as f:
        meta = json.load(f)
        return meta["cost"]


def run_simulation(config_path=None, nx=40, ny=40, verbose=False, **kwargs):
    """
    Run plate with hole simulation with given parameters.

    Args:
        config_path: Path to YAML config file (optional)
        nx: Number of elements in x direction (tunable parameter)
        ny: Number of elements in y direction (tunable parameter)
        verbose: Enable verbose output
        **kwargs: Additional parameters to override

    Returns:
        dict: Simulation results including error metric
    """
    runner_path = _find_runner_path()

    if config_path:
        cmd = ["PYTHONPATH=/home/yadi/costsci-tools", "python", runner_path, f"--config-path={config_path}"]
    else:
        cmd = ["PYTHONPATH=/home/yadi/costsci-tools", "python", runner_path]

    # Override parameters
    cmd.extend([f"nx={nx}", f"ny={ny}", f"verbose={verbose}"])

    for key, value in kwargs.items():
        cmd.append(f"{key}={value}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print("STDOUT:", result.stdout)
        return {"success": True, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": str(e), "stdout": e.stdout, "stderr": e.stderr}


def load_results(sim_dir, frame=0):
    """
    Load simulation results from output directory.

    Args:
        sim_dir: Simulation output directory
        frame: Frame number to load (default: 0 for static problems)

    Returns:
        dict: Loaded simulation results
    """
    h5_file = os.path.join(sim_dir, f"frame_{frame:04d}.h5")
    json_file = os.path.join(sim_dir, f"frame_{frame:04d}.json")

    results = {}

    if os.path.exists(h5_file):
        with h5py.File(h5_file, 'r') as f:
            results['displacement'] = f['displacement'][:]
            results['coordinates'] = f['coordinates'][:]
            results['elements'] = f['elements'][:]
            results['error'] = f.attrs['error']
            results['nx'] = f.attrs['nx']
            results['ny'] = f.attrs['ny']

    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            results.update(json_data)

    return results


def get_error_metric(sim_dir, frame=0):
    """
    Extract error metric from simulation results.

    Args:
        sim_dir: Simulation output directory
        frame: Frame number (default: 0)

    Returns:
        float: Normalized stress error
    """
    results = load_results(sim_dir, frame)
    return results.get('error', None)


def compare_convergence(sim_dirs, labels=None, save_path=None):
    """
    Compare convergence across different simulations.

    Args:
        sim_dirs: List of simulation directories
        labels: Labels for each simulation
        save_path: Path to save comparison plot

    Returns:
        dict: Comparison results
    """
    if labels is None:
        labels = [f"Sim {i+1}" for i in range(len(sim_dirs))]

    errors = []
    nx_values = []

    for sim_dir in sim_dirs:
        results = load_results(sim_dir)
        if results:
            errors.append(results.get('error', np.nan))
            nx_values.append(results.get('nx', np.nan))
        else:
            errors.append(np.nan)
            nx_values.append(np.nan)

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(nx_values, errors, 'o-', linewidth=2, markersize=8)

    for i, (nx, err, label) in enumerate(zip(nx_values, errors, labels)):
        if not np.isnan(err):
            ax.annotate(label, (nx, err), xytext=(5, 5),
                       textcoords='offset points', fontsize=10)

    ax.set_xlabel('Number of Elements (nx)')
    ax.set_ylabel('Normalized Stress Error')
    ax.set_title('Plate with Hole: Convergence Study')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {
        'nx_values': nx_values,
        'errors': errors,
        'labels': labels
    }


def check_success_criteria(sim_dir, target_error=1e-3, frame=0):
    """
    Check if simulation meets success criteria.

    Args:
        sim_dir: Simulation directory
        target_error: Target error threshold
        frame: Frame number

    Returns:
        dict: Success check results
    """
    try:
        error = get_error_metric(sim_dir, frame)
        if error is None:
            return {"success": False, "reason": "Could not extract error metric"}

        success = error <= target_error
        return {
            "success": success,
            "error": error,
            "target_error": target_error,
            "reason": f"Error {error:.2e} {'<=' if success else '>'} target {target_error:.2e}"
        }
    except Exception as e:
        return {"success": False, "reason": f"Error checking results: {str(e)}"}


def run_parameter_sweep(nx_values, ny_values=None, config_path=None, **kwargs):
    """
    Run parameter sweep over mesh resolutions.

    Args:
        nx_values: List of nx values to test
        ny_values: List of ny values (if None, uses nx_values)
        config_path: Base config path
        **kwargs: Additional parameters

    Returns:
        dict: Sweep results
    """
    if ny_values is None:
        ny_values = nx_values

    results = []

    for nx, ny in zip(nx_values, ny_values):
        print(f"Running simulation with nx={nx}, ny={ny}")

        sim_result = run_simulation(
            config_path=config_path,
            nx=nx,
            ny=ny,
            **kwargs
        )

        if sim_result["success"]:
            # Extract simulation directory from output
            sim_dir = f"sim_res/plate_with_a_hole/p1_nx_{nx}_ny_{ny}"
            error = get_error_metric(sim_dir)

            results.append({
                "nx": nx,
                "ny": ny,
                "error": error,
                "sim_dir": sim_dir,
                "success": True
            })
        else:
            results.append({
                "nx": nx,
                "ny": ny,
                "error": None,
                "sim_dir": None,
                "success": False,
                "error_msg": sim_result.get("error", "Unknown error")
            })

    return {"sweep_results": results}


# # Default parameter configurations based on CSV metadata
# DEFAULT_CONFIGS = {
#     "fine": {"nx": 80, "ny": 80},      # Target error ~1e-3
#     "medium": {"nx": 40, "ny": 40},    # Target error ~5e-3
#     "coarse": {"nx": 20, "ny": 20},    # Target error ~1e-2
# }