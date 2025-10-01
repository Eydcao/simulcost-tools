import os
import subprocess
import h5py
import numpy as np
import json
import matplotlib.pyplot as plt


def _find_runner_path():
    """Automatically find the correct path to hasegawa_mima_linear.py runner."""
    cwd = os.getcwd()

    possible_paths = []

    if cwd.endswith('SimulCost-Bench'):
        possible_paths.extend([
            "costsci_tools/runners/hasegawa_mima_linear.py",
            "runners/hasegawa_mima_linear.py"
        ])
    elif cwd.endswith('costsci_tools') or 'costsci_tools' in cwd:
        possible_paths.extend([
            "runners/hasegawa_mima_linear.py",
            "../runners/hasegawa_mima_linear.py",
            "costsci_tools/runners/hasegawa_mima_linear.py"
        ])
    else:
        possible_paths.extend([
            "runners/hasegawa_mima_linear.py",
            "costsci_tools/runners/hasegawa_mima_linear.py",
            "./runners/hasegawa_mima_linear.py"
        ])

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Could not find hasegawa_mima_linear.py runner. Searched: {possible_paths}")


def run_sim_hasegawa_mima_linear(profile, N, dt, cg_atol, analytical):
    """Run the Hasegawa-Mima linear simulation with the given parameters if not already simulated."""
    if analytical:
        method_suffix = "_analytical"
        dir_path = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}" + method_suffix + "/"
    else:
        method_suffix = "_numerical"
        dir_path = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}_cg_{cg_atol:.2e}" + method_suffix + "/"
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
    cmd = f"PYTHONPATH=/home/yadi/costsci-tools python {runner_path} --config-name={profile} N={N} dt={dt} cg_atol={cg_atol} analytical={analytical}"
    subprocess.run(cmd, shell=True, check=True)

    # Read the meta.json to get cost
    with open(meta_path, "r") as f:
        meta = json.load(f)
        return meta["cost"]


# def run_sim_hasegawa_mima_linear_analytical(profile, N):
#     """Run analytical solution (dt and cg_atol don't matter for analytical)"""
#     return run_sim_hasegawa_mima_linear(profile, N, dt=1.0, analytical=True)


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
        cmd = ["PYTHONPATH=/home/yadi/costsci-tools", "python", runner_path, f"--config-path={config_path}"]
    else:
        cmd = ["PYTHONPATH=/home/yadi/costsci-tools", "python", runner_path]

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
        with h5py.File(h5_file, 'r') as f:
            results['phi'] = f['phi'][:]
            results['coordinates_x'] = f['coordinates_x'][:]
            results['coordinates_y'] = f['coordinates_y'][:]
            results['time'] = f.attrs['time']
            results['N'] = f.attrs['N']
            results['dt'] = f.attrs['dt']

    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
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
    # Load metadata to get analytical reference directory
    meta_file = os.path.join(numerical_sim_dir, "meta.json")
    if not os.path.exists(meta_file):
        print(f"Warning: meta.json not found in {numerical_sim_dir}")
        return None

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    # Check if this is an analytical run (no error needed)
    if meta.get('analytical', False):
        return 0.0

    # Get analytical reference directory
    analytical_sim_dir = meta.get('analytical_reference_dir', None)
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
    frame_files = sorted([f for f in os.listdir(numerical_sim_dir) if f.startswith('frame_') and f.endswith('.h5')])

    for frame_file in frame_files:
        frame_num = int(frame_file.split('_')[1].split('.')[0])

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
        phi_num = np.array(num_res['phi'])
        phi_ana = np.array(ana_res['phi'])
        diff = phi_num - phi_ana

        l2_error = np.sqrt(np.mean(diff**2))

        l2_errors.append(l2_error)
        times.append(num_res['time'])

    # Note: Plotting removed from wrapper - visualization happens in solver's dump() method

    return {
        "success": True,
        "times": times,
        "l2_errors": l2_errors,
        "mean_l2_error": np.mean(l2_errors),
        "max_l2_error": np.max(l2_errors)
    }


# def check_success_criteria(sim_dir, target_error):
#     """
#     Check if simulation meets success criteria.

#     Args:
#         sim_dir: Simulation directory
#         target_error: Target error threshold

#     Returns:
#         dict: Success check results
#     """
#     try:
#         error = get_error_metric(sim_dir)
#         if error is None:
#             return {"success": False, "reason": "Could not extract error metric"}

#         success = error <= target_error
#         return {
#             "success": success,
#             "error": error,
#             "target_error": target_error,
#             "reason": f"Error {error:.2e} {'<=' if success else '>'} target {target_error:.2e}"
#         }
#     except Exception as e:
#         return {"success": False, "reason": f"Error checking results: {str(e)}"}


# def run_parameter_sweep(N_values, dt_values, cg_atol_values, config_path, **kwargs):
#     """
#     Run parameter sweep over numerical parameters.

#     Args:
#         N_values: List of N values to test
#         dt_values: List of dt values to test
#         cg_atol_values: List of CG tolerance values (optional)
#         config_path: Base config path
#         **kwargs: Additional parameters

#     Returns:
#         dict: Sweep results
#     """
#     if cg_atol_values is None:
#         cg_atol_values = [1e-6]

#     results = []

#     for N in N_values:
#         for dt in dt_values:
#             for cg_atol in cg_atol_values:
#                 print(f"Running simulation with N={N}, dt={dt}, cg_atol={cg_atol:.2e}")

#                 sim_result = run_simulation(
#                     config_path=config_path,
#                     N=N,
#                     dt=dt,
#                     cg_atol=cg_atol,
#                     analytical=False,
#                     **kwargs
#                 )

#                 if sim_result["success"]:
#                     # Extract simulation directory from parameters
#                     sim_dir = f"sim_res/hasegawa_mima_linear/p1_N_{N}_dt_{dt:.2e}_cg_{cg_atol:.2e}_numerical"
#                     error = get_error_metric(sim_dir)

#                     results.append({
#                         "N": N,
#                         "dt": dt,
#                         "cg_atol": cg_atol,
#                         "error": error,
#                         "sim_dir": sim_dir,
#                         "success": True
#                     })
#                 else:
#                     results.append({
#                         "N": N,
#                         "dt": dt,
#                         "cg_atol": cg_atol,
#                         "error": None,
#                         "sim_dir": None,
#                         "success": False,
#                         "error_msg": sim_result.get("error", "Unknown error")
#                     })

#     return {"sweep_results": results}


# # Default parameter configurations for different accuracy/cost targets
# DEFAULT_CONFIGS = {
#     "fast": {"N": 64, "dt": 50.0, "cg_atol": 1e-4},      # Fast but less accurate
#     "balanced": {"N": 128, "dt": 20.0, "cg_atol": 1e-5},  # Balanced accuracy/cost
#     "accurate": {"N": 256, "dt": 10.0, "cg_atol": 1e-6},  # High accuracy
# }