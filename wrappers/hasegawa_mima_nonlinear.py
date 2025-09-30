import os
import subprocess
import h5py
import numpy as np
import json
import matplotlib.pyplot as plt


def _find_runner_path():
    """Automatically find the correct path to hasegawa_mima_nonlinear.py runner."""
    cwd = os.getcwd()

    possible_paths = []

    if cwd.endswith('SimulCost-Bench'):
        possible_paths.extend([
            "costsci_tools/runners/hasegawa_mima_nonlinear.py",
            "runners/hasegawa_mima_nonlinear.py"
        ])
    elif cwd.endswith('costsci_tools') or 'costsci_tools' in cwd:
        possible_paths.extend([
            "runners/hasegawa_mima_nonlinear.py",
            "../runners/hasegawa_mima_nonlinear.py",
            "costsci_tools/runners/hasegawa_mima_nonlinear.py"
        ])
    else:
        possible_paths.extend([
            "runners/hasegawa_mima_nonlinear.py",
            "costsci_tools/runners/hasegawa_mima_nonlinear.py",
            "./runners/hasegawa_mima_nonlinear.py"
        ])

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Could not find hasegawa_mima_nonlinear.py runner. Searched: {possible_paths}")


def run_sim_hasegawa_mima_nonlinear(profile, N, dt):
    """Run the Hasegawa-Mima nonlinear simulation with the given parameters if not already simulated."""
    dir_path = f"sim_res/hasegawa_mima_nonlinear/{profile}_N_{N}_dt_{dt:.2e}_nonlinear/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                return meta["cost"]

    # Run the simulation if not already done
    print(f"Running new nonlinear simulation with parameters: N={N}, dt={dt}")
    runner_path = _find_runner_path()
    cmd = f"PYTHONPATH=/home/yadi/costsci-tools python {runner_path} --config-name={profile} N={N} dt={dt}"
    subprocess.run(cmd, shell=True, check=True)

    # Read the meta.json to get cost
    with open(meta_path, "r") as f:
        meta = json.load(f)
        return meta["cost"]


def run_simulation(config_path=None, N=128, dt=10.0, verbose=False, **kwargs):
    """
    Run Hasegawa-Mima nonlinear simulation with given parameters.

    Args:
        config_path: Path to YAML config file (optional)
        N: Grid resolution (main tunable parameter)
        dt: Time step (main tunable parameter)
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
    cmd.extend([f"N={N}", f"dt={dt}", f"verbose={verbose}"])

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
        frame: Frame number to load (default: 0)

    Returns:
        dict: Loaded simulation results
    """
    h5_file = os.path.join(sim_dir, f"frame_{frame:04d}.h5")

    results = {}

    if os.path.exists(h5_file):
        with h5py.File(h5_file, 'r') as f:
            results['phi'] = f['phi'][:]
            results['coordinates_x'] = f['coordinates_x'][:]
            results['coordinates_y'] = f['coordinates_y'][:]
            results['time'] = f.attrs['time']
            results['N'] = f.attrs['N']
            results['dt'] = f.attrs['dt']
            results['dealias_ratio'] = f.attrs['dealias_ratio']

    return results


def get_cost_metric(sim_dir):
    """
    Extract cost metric from simulation results.

    Args:
        sim_dir: Simulation output directory

    Returns:
        float: Computational cost estimate
    """
    meta_file = os.path.join(sim_dir, "meta.json")
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            return meta.get('cost', None)
    return None


def compare_resolutions(coarse_sim_dir, fine_sim_dir, save_path=None):
    """
    Compare solutions at different resolutions for convergence checking.

    Args:
        coarse_sim_dir: Directory with coarse resolution results
        fine_sim_dir: Directory with fine resolution results
        save_path: Path to save comparison plot

    Returns:
        dict: Comparison results including error metrics
    """
    # Load both solutions
    coarse_results = []
    fine_results = []

    # Find all frame files in coarse directory
    frame_files = sorted([f for f in os.listdir(coarse_sim_dir) if f.startswith('frame_') and f.endswith('.h5')])

    for frame_file in frame_files:
        frame_num = int(frame_file.split('_')[1].split('.')[0])

        coarse_result = load_results(coarse_sim_dir, frame_num)
        fine_result = load_results(fine_sim_dir, frame_num)

        if coarse_result and fine_result:
            coarse_results.append(coarse_result)
            fine_results.append(fine_result)

    if not coarse_results:
        return {"success": False, "reason": "No valid results found"}

    # Calculate error metrics between resolutions (L2 norm only)
    l2_errors = []
    times = []

    for coarse_res, fine_res in zip(coarse_results, fine_results):
        # Interpolate fine solution to coarse grid for comparison
        phi_coarse = np.array(coarse_res['phi'])
        phi_fine = np.array(fine_res['phi'])

        # Simple downsampling for comparison (could use interpolation)
        N_coarse = phi_coarse.shape[0]
        N_fine = phi_fine.shape[0]
        step = N_fine // N_coarse
        phi_fine_downsampled = phi_fine[::step, ::step]

        diff = phi_coarse - phi_fine_downsampled

        l2_error = np.sqrt(np.mean(diff**2))

        l2_errors.append(l2_error)
        times.append(coarse_res['time'])

    # Note: Plotting removed from wrapper - visualization happens in solver's dump() method

    return {
        "success": True,
        "times": times,
        "l2_errors": l2_errors,
        "mean_l2_error": np.mean(l2_errors),
        "max_l2_error": np.max(l2_errors)
    }


def check_convergence_criteria(coarse_sim_dir, fine_sim_dir, target_error=1e-3):
    """
    Check if simulation meets convergence criteria by comparing with higher resolution.

    Args:
        coarse_sim_dir: Coarse resolution simulation directory
        fine_sim_dir: Fine resolution simulation directory
        target_error: Target error threshold for convergence

    Returns:
        dict: Convergence check results
    """
    try:
        comparison = compare_resolutions(coarse_sim_dir, fine_sim_dir)
        if not comparison["success"]:
            return {"success": False, "reason": comparison["reason"]}

        error = comparison["mean_l2_error"]
        success = error <= target_error

        return {
            "success": success,
            "error": error,
            "target_error": target_error,
            "reason": f"Error {error:.2e} {'<=' if success else '>'} target {target_error:.2e}"
        }
    except Exception as e:
        return {"success": False, "reason": f"Error checking convergence: {str(e)}"}


def run_parameter_sweep(N_values, dt_values, config_path=None, **kwargs):
    """
    Run parameter sweep over numerical parameters.

    Args:
        N_values: List of N values to test
        dt_values: List of dt values to test
        config_path: Base config path
        **kwargs: Additional parameters

    Returns:
        dict: Sweep results
    """
    results = []

    for N in N_values:
        for dt in dt_values:
            print(f"Running simulation with N={N}, dt={dt}")

            sim_result = run_simulation(
                config_path=config_path,
                N=N,
                dt=dt,
                **kwargs
            )

            if sim_result["success"]:
                # Extract simulation directory from parameters
                sim_dir = f"sim_res/hasegawa_mima_nonlinear/p1_N_{N}_dt_{dt:.2e}_nonlinear"
                cost = get_cost_metric(sim_dir)

                results.append({
                    "N": N,
                    "dt": dt,
                    "cost": cost,
                    "sim_dir": sim_dir,
                    "success": True
                })
            else:
                results.append({
                    "N": N,
                    "dt": dt,
                    "cost": None,
                    "sim_dir": None,
                    "success": False,
                    "error_msg": sim_result.get("error", "Unknown error")
                })

    return {"sweep_results": results}


def run_convergence_study(base_N=64, N_multiplier=2, max_levels=3, dt=10.0, config_path=None, **kwargs):
    """
    Run convergence study by comparing solutions at different resolutions.

    Args:
        base_N: Base resolution
        N_multiplier: Resolution multiplication factor
        max_levels: Maximum number of resolution levels
        dt: Time step (kept constant for resolution study)
        config_path: Base config path
        **kwargs: Additional parameters

    Returns:
        dict: Convergence study results
    """
    N_values = [base_N * (N_multiplier ** i) for i in range(max_levels)]

    print(f"Running convergence study with N values: {N_values}")

    # Run simulations at all resolutions
    sweep_results = run_parameter_sweep(N_values, [dt], config_path, **kwargs)

    if not sweep_results["success"]:
        return sweep_results

    # Compare adjacent resolution levels
    convergence_results = []

    for i in range(len(N_values) - 1):
        N_coarse = N_values[i]
        N_fine = N_values[i + 1]

        coarse_dir = f"sim_res/hasegawa_mima_nonlinear/p1_N_{N_coarse}_dt_{dt:.2e}_nonlinear"
        fine_dir = f"sim_res/hasegawa_mima_nonlinear/p1_N_{N_fine}_dt_{dt:.2e}_nonlinear"

        comparison = compare_resolutions(coarse_dir, fine_dir)

        if comparison["success"]:
            convergence_results.append({
                "N_coarse": N_coarse,
                "N_fine": N_fine,
                "error": comparison["mean_l2_error"],
                "max_error": comparison["max_l2_error"]
            })

    return {
        "success": True,
        "N_values": N_values,
        "convergence_results": convergence_results,
        "sweep_results": sweep_results
    }


# # Default parameter configurations for different accuracy/cost targets
# DEFAULT_CONFIGS = {
#     "fast": {"N": 64, "dt": 20.0},      # Fast but coarse resolution
#     "balanced": {"N": 128, "dt": 10.0}, # Balanced accuracy/cost
#     "accurate": {"N": 256, "dt": 5.0},  # High accuracy
# }