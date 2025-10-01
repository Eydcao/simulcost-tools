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


def get_results(profile, N, dt):
    """
    Get simulation results by running the simulation (if needed) and loading all frames.

    Args:
        profile: Configuration profile name
        N: Grid resolution
        dt: Time step size

    Returns:
        tuple: (cost, results_list) where results_list contains all frame data
    """
    dir_path = f"sim_res/hasegawa_mima_nonlinear/{profile}_N_{N}_dt_{dt:.2e}_nonlinear/"
    meta_path = os.path.join(dir_path, "meta.json")

    # Check if the simulation has already been run
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta:
                print(f"Using existing simulation results from {dir_path}")
                cost = meta["cost"]
    else:
        # Run the simulation if not already done
        print(f"Running new nonlinear simulation with parameters: N={N}, dt={dt}")
        runner_path = _find_runner_path()
        cmd = f"PYTHONPATH=/home/yadi/costsci-tools python {runner_path} --config-name={profile} N={N} dt={dt}"
        subprocess.run(cmd, shell=True, check=True)

        # Read the meta.json to get cost
        with open(meta_path, "r") as f:
            meta = json.load(f)
            cost = meta["cost"]

    # Load all frames
    results_list = []
    frame_files = sorted([f for f in os.listdir(dir_path) if f.startswith('frame_') and f.endswith('.h5')])

    for frame_file in frame_files:
        frame_num = int(frame_file.split('_')[1].split('.')[0])
        h5_file = os.path.join(dir_path, frame_file)

        with h5py.File(h5_file, 'r') as f:
            result = {
                'phi': f['phi'][:],
                'coordinates_x': f['coordinates_x'][:],
                'coordinates_y': f['coordinates_y'][:],
                'time': f.attrs['time'],
                'N': f.attrs['N'],
                'dt': f.attrs['dt'],
                'dealias_ratio': f.attrs['dealias_ratio']
            }
            results_list.append(result)

    return cost, results_list

def compare_solutions(profile, params1, params2, tolerance_rmse):
    """
    Compare two Hasegawa-Mima nonlinear simulations to check for convergence.

    Args:
        profile: Configuration profile name
        params1: Dictionary with first simulation parameters (coarse)
        params2: Dictionary with second simulation parameters (fine)
        tolerance_rmse: RMSE tolerance for convergence

    Returns:
        is_converged: Boolean indicating if solutions converged
        cost1: Cost for first simulation
        cost2: Cost for second simulation
        rmse_diff: RMSE difference between solutions
    """
    # Get both simulation results
    cost1, results1 = get_results(profile, **params1)
    cost2, results2 = get_results(profile, **params2)

    if not results1 or not results2:
        return False, cost1, cost2, None

    # Calculate L2 error between resolutions for all frames
    l2_errors = []

    for res1, res2 in zip(results1, results2):
        # Get phi fields
        phi_coarse = np.array(res1['phi'])
        phi_fine = np.array(res2['phi'])

        # Downsample fine solution to coarse grid for comparison
        N_coarse = phi_coarse.shape[0]
        N_fine = phi_fine.shape[0]
        step = N_fine // N_coarse
        phi_fine_downsampled = phi_fine[::step, ::step]

        # Calculate L2 error
        diff = phi_coarse - phi_fine_downsampled
        l2_error = np.sqrt(np.mean(diff**2))
        l2_errors.append(l2_error)

    # Use mean L2 error across all frames
    rmse_diff = np.mean(l2_errors)

    # Check if error between resolutions is below tolerance
    is_converged = rmse_diff <= tolerance_rmse

    return is_converged, cost1, cost2, rmse_diff
