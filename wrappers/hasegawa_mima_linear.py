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


def run_sim_hasegawa_mima_linear(profile, N, dt, cg_atol=1e-6, analytical=False):
    """Run the Hasegawa-Mima linear simulation with the given parameters if not already simulated."""
    method_suffix = "_analytical" if analytical else "_numerical"
    dir_path = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}" + method_suffix + "/"
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


def run_sim_hasegawa_mima_linear_analytical(profile, N):
    """Run analytical solution (dt and cg_atol don't matter for analytical)"""
    return run_sim_hasegawa_mima_linear(profile, N, dt=1.0, analytical=True)


def run_simulation(config_path=None, N=256, dt=20.0, cg_atol=1e-6, analytical=False, verbose=False, **kwargs):
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
            results['analytical'] = f.attrs['analytical']

    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            results.update(json_data)

    return results


def get_error_metric(sim_dir):
    """
    Extract error metric from simulation results.

    Args:
        sim_dir: Simulation output directory

    Returns:
        float: Error compared to analytical solution
    """
    meta_file = os.path.join(sim_dir, "meta.json")
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            return meta.get('error', None)
    return None


def compare_with_analytical(numerical_sim_dir, analytical_sim_dir, save_path=None):
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

    # Calculate error metrics
    l2_errors = []
    linf_errors = []
    times = []

    for num_res, ana_res in zip(numerical_results, analytical_results):
        phi_num = np.array(num_res['phi'])
        phi_ana = np.array(ana_res['phi'])
        diff = phi_num - phi_ana

        l2_error = np.sqrt(np.mean(diff**2))
        linf_error = np.max(np.abs(diff))

        l2_errors.append(l2_error)
        linf_errors.append(linf_error)
        times.append(num_res['time'])

    # Create comparison visualization
    if save_path or len(numerical_results) > 0:
        n_times = len(numerical_results)
        fig, axes = plt.subplots(n_times, 3, figsize=(15, 5*n_times))

        if n_times == 1:
            axes = axes.reshape(1, -1)

        # Get coordinates
        x = numerical_results[0]['coordinates_x']
        y = numerical_results[0]['coordinates_y']
        X, Y = np.meshgrid(x, y)

        # Find common color limits
        all_phi = []
        for num_res, ana_res in zip(numerical_results, analytical_results):
            all_phi.extend([num_res['phi'], ana_res['phi']])
        vmin, vmax = np.min(all_phi), np.max(all_phi)

        for i, (num_res, ana_res) in enumerate(zip(numerical_results, analytical_results)):
            phi_num = np.array(num_res['phi'])
            phi_ana = np.array(ana_res['phi'])
            diff = phi_num - phi_ana
            t = times[i]

            # Numerical solution
            im1 = axes[i, 0].pcolormesh(X, Y, phi_num, cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f'Numerical t={t:.1f}')
            axes[i, 0].set_xlabel('x')
            axes[i, 0].set_ylabel('y')
            plt.colorbar(im1, ax=axes[i, 0])

            # Analytical solution
            im2 = axes[i, 1].pcolormesh(X, Y, phi_ana, cmap='RdBu', shading='auto', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'Analytical t={t:.1f}')
            axes[i, 1].set_xlabel('x')
            axes[i, 1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[i, 1])

            # Difference
            diff_max = np.max(np.abs(diff))
            im3 = axes[i, 2].pcolormesh(X, Y, diff, cmap='bwr', shading='auto',
                                       vmin=-diff_max, vmax=diff_max)
            axes[i, 2].set_title(f'Difference\nL2={l2_errors[i]:.2e}, L∞={linf_errors[i]:.2e}')
            axes[i, 2].set_xlabel('x')
            axes[i, 2].set_ylabel('y')
            plt.colorbar(im3, ax=axes[i, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return {
        "success": True,
        "times": times,
        "l2_errors": l2_errors,
        "linf_errors": linf_errors,
        "mean_l2_error": np.mean(l2_errors),
        "mean_linf_error": np.mean(linf_errors),
        "max_l2_error": np.max(l2_errors),
        "max_linf_error": np.max(linf_errors)
    }


def check_success_criteria(sim_dir, target_error=1e-4):
    """
    Check if simulation meets success criteria.

    Args:
        sim_dir: Simulation directory
        target_error: Target error threshold

    Returns:
        dict: Success check results
    """
    try:
        error = get_error_metric(sim_dir)
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


def run_parameter_sweep(N_values, dt_values, cg_atol_values=None, config_path=None, **kwargs):
    """
    Run parameter sweep over numerical parameters.

    Args:
        N_values: List of N values to test
        dt_values: List of dt values to test
        cg_atol_values: List of CG tolerance values (optional)
        config_path: Base config path
        **kwargs: Additional parameters

    Returns:
        dict: Sweep results
    """
    if cg_atol_values is None:
        cg_atol_values = [1e-6]

    results = []

    for N in N_values:
        for dt in dt_values:
            for cg_atol in cg_atol_values:
                print(f"Running simulation with N={N}, dt={dt}, cg_atol={cg_atol:.2e}")

                sim_result = run_simulation(
                    config_path=config_path,
                    N=N,
                    dt=dt,
                    cg_atol=cg_atol,
                    analytical=False,
                    **kwargs
                )

                if sim_result["success"]:
                    # Extract simulation directory from parameters
                    sim_dir = f"sim_res/hasegawa_mima_linear/p1_N_{N}_dt_{dt:.2e}_numerical"
                    error = get_error_metric(sim_dir)

                    results.append({
                        "N": N,
                        "dt": dt,
                        "cg_atol": cg_atol,
                        "error": error,
                        "sim_dir": sim_dir,
                        "success": True
                    })
                else:
                    results.append({
                        "N": N,
                        "dt": dt,
                        "cg_atol": cg_atol,
                        "error": None,
                        "sim_dir": None,
                        "success": False,
                        "error_msg": sim_result.get("error", "Unknown error")
                    })

    return {"sweep_results": results}


# Default parameter configurations for different accuracy/cost targets
DEFAULT_CONFIGS = {
    "fast": {"N": 64, "dt": 50.0, "cg_atol": 1e-4},      # Fast but less accurate
    "balanced": {"N": 128, "dt": 20.0, "cg_atol": 1e-5},  # Balanced accuracy/cost
    "accurate": {"N": 256, "dt": 10.0, "cg_atol": 1e-6},  # High accuracy
}