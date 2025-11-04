#!/usr/bin/env python3
"""
Script to measure maximum wall time for the most computationally expensive cases
in hasegawa_mima_linear and hasegawa_mima_nonlinear datasets.

For each profile, finds the case with:
- Maximum N (spatial resolution)
- Minimum dt (time step)
Then runs the simulation and measures wall time.
"""

import json
import time
import subprocess
import sys
import os
from collections import defaultdict
from pathlib import Path

# Python interpreter path
PYTHON_PATH = "/home/ubuntu/miniconda3/envs/casebench/bin/python"


def load_successful_tasks(dataset_path):
    """Load successful tasks from JSON file."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data['tasks']


def find_max_complexity_per_profile(tasks, solver_type):
    """
    Find the maximum complexity case for each profile.
    Max complexity = largest N and smallest dt

    Returns: dict[profile] -> dict with max_N case and min_dt case
    """
    profile_cases = defaultdict(lambda: {
        'max_N_case': None,
        'max_N': 0,
        'min_dt_case': None,
        'min_dt': float('inf')
    })

    for task in tasks:
        profile = task['profile']

        # Get the optimal parameter value and non-target parameters
        optimal_value = task['results']['optimal_parameter_value']
        target_param = task['target_parameter']
        non_target = task['non_target_parameters']

        # Reconstruct the full parameter set
        if target_param == 'N':
            N = optimal_value
            dt = non_target.get('dt')
        elif target_param == 'dt':
            N = non_target.get('N')
            dt = optimal_value
        else:
            # For cg_atol target, get N and dt from non-target
            N = non_target.get('N')
            dt = non_target.get('dt')

        # Track max N case
        if N is not None and N > profile_cases[profile]['max_N']:
            profile_cases[profile]['max_N'] = N
            profile_cases[profile]['max_N_case'] = {
                'N': N,
                'dt': dt,
                'cg_atol': non_target.get('cg_atol') if solver_type == 'linear' else None,
                'task': task
            }

        # Track min dt case
        if dt is not None and dt < profile_cases[profile]['min_dt']:
            profile_cases[profile]['min_dt'] = dt
            profile_cases[profile]['min_dt_case'] = {
                'N': N,
                'dt': dt,
                'cg_atol': non_target.get('cg_atol') if solver_type == 'linear' else None,
                'task': task
            }

    return dict(profile_cases)


def run_simulation_with_timing(solver_type, profile, N, dt, cg_atol=None):
    """
    Run a simulation and measure wall time.

    Args:
        solver_type: 'linear' or 'nonlinear'
        profile: profile name (e.g., 'p1', 'p2')
        N: spatial resolution
        dt: time step
        cg_atol: CG tolerance (for linear solver only)

    Returns:
        dict with wall_time, cost, and other metadata
    """
    # Determine runner path
    if solver_type == 'linear':
        runner_path = 'runners/hasegawa_mima_linear.py'
        if cg_atol is None:
            cg_atol = 1e-8  # default
        cmd = [
            PYTHON_PATH, runner_path,
            f'--config-name={profile}',
            f'N={N}',
            f'dt={dt}',
            f'cg_atol={cg_atol}',
            'analytical=false',
            'verbose=false'
        ]
    else:  # nonlinear
        runner_path = 'runners/hasegawa_mima_nonlinear.py'
        cmd = [
            PYTHON_PATH, runner_path,
            f'--config-name={profile}',
            f'N={N}',
            f'dt={dt}',
            'verbose=false'
        ]

    print(f"\n{'='*80}")
    print(f"Running {solver_type} simulation:")
    print(f"  Profile: {profile}")
    print(f"  N: {N}")
    print(f"  dt: {dt}")
    if cg_atol is not None:
        print(f"  cg_atol: {cg_atol}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*80}")

    # Measure wall time
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd()
        )
        wall_time = time.time() - start_time
        success = True
        error_msg = None

        print(f"✅ Simulation completed successfully in {wall_time:.2f} seconds")

    except subprocess.CalledProcessError as e:
        wall_time = time.time() - start_time
        success = False
        error_msg = str(e)

        print(f"❌ Simulation failed after {wall_time:.2f} seconds")
        print(f"Error: {error_msg}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")

    # Try to read cost from meta.json
    if solver_type == 'linear':
        if cg_atol is not None:
            meta_path = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}_cg_{cg_atol:.2e}_numerical/meta.json"
        else:
            meta_path = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}_numerical/meta.json"
    else:
        meta_path = f"sim_res/hasegawa_mima_nonlinear/{profile}_N_{N}_dt_{dt:.3g}_nonlinear/meta.json"

    cost = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                cost = meta.get('cost')
        except Exception as e:
            print(f"Warning: Could not read cost from {meta_path}: {e}")

    return {
        'wall_time': wall_time,
        'cost': cost,
        'success': success,
        'error_msg': error_msg,
        'N': N,
        'dt': dt,
        'cg_atol': cg_atol
    }


def generate_markdown_report(linear_results, nonlinear_results, output_path):
    """Generate markdown report with timing results."""

    with open(output_path, 'w') as f:
        f.write("# Maximum Wall Time Measurements\n\n")
        f.write("This report shows wall time measurements for the most computationally expensive cases ")
        f.write("in the Hasegawa-Mima linear and nonlinear datasets.\n\n")
        f.write("For each profile, we measure:\n")
        f.write("- **Max N case**: Largest spatial resolution\n")
        f.write("- **Min dt case**: Smallest time step\n\n")

        # Linear results
        f.write("## Hasegawa-Mima Linear\n\n")
        f.write("| Profile | Case Type | N | dt | cg_atol | Wall Time (s) | Cost | Status |\n")
        f.write("|---------|-----------|---|----|---------|--------------:|-----:|--------|\n")

        for profile in sorted(linear_results.keys()):
            for case_type in ['max_N_case', 'min_dt_case']:
                result = linear_results[profile].get(case_type)
                if result:
                    case_label = "Max N" if case_type == 'max_N_case' else "Min dt"
                    N = result['N']
                    dt = result['dt']
                    cg_atol = result.get('cg_atol', 'N/A')
                    wall_time = result.get('wall_time', 'N/A')
                    cost = result.get('cost', 'N/A')
                    status = '✅' if result.get('success', False) else '❌'

                    if isinstance(wall_time, (int, float)):
                        wall_time_str = f"{wall_time:.2f}"
                    else:
                        wall_time_str = str(wall_time)

                    if isinstance(cost, (int, float)):
                        cost_str = f"{cost:.2e}"
                    else:
                        cost_str = str(cost)

                    if isinstance(cg_atol, (int, float)):
                        cg_atol_str = f"{cg_atol:.1e}"
                    else:
                        cg_atol_str = str(cg_atol)

                    f.write(f"| {profile} | {case_label} | {N} | {dt} | {cg_atol_str} | {wall_time_str} | {cost_str} | {status} |\n")

        # Nonlinear results
        f.write("\n## Hasegawa-Mima Nonlinear\n\n")
        f.write("| Profile | Case Type | N | dt | Wall Time (s) | Cost | Status |\n")
        f.write("|---------|-----------|---|----|--------------:|-----:|--------|\n")

        for profile in sorted(nonlinear_results.keys()):
            for case_type in ['max_N_case', 'min_dt_case']:
                result = nonlinear_results[profile].get(case_type)
                if result:
                    case_label = "Max N" if case_type == 'max_N_case' else "Min dt"
                    N = result['N']
                    dt = result['dt']
                    wall_time = result.get('wall_time', 'N/A')
                    cost = result.get('cost', 'N/A')
                    status = '✅' if result.get('success', False) else '❌'

                    if isinstance(wall_time, (int, float)):
                        wall_time_str = f"{wall_time:.2f}"
                    else:
                        wall_time_str = str(wall_time)

                    if isinstance(cost, (int, float)):
                        cost_str = f"{cost:.2e}"
                    else:
                        cost_str = str(cost)

                    f.write(f"| {profile} | {case_label} | {N} | {dt} | {wall_time_str} | {cost_str} | {status} |\n")

        # Summary statistics
        f.write("\n## Summary Statistics\n\n")

        # Linear summary
        linear_times = []
        for profile_data in linear_results.values():
            for case_data in [profile_data.get('max_N_case'), profile_data.get('min_dt_case')]:
                if case_data and isinstance(case_data.get('wall_time'), (int, float)):
                    linear_times.append(case_data['wall_time'])

        if linear_times:
            f.write(f"**Linear solver:**\n")
            f.write(f"- Total runs: {len(linear_times)}\n")
            f.write(f"- Max wall time: {max(linear_times):.2f} s\n")
            f.write(f"- Min wall time: {min(linear_times):.2f} s\n")
            f.write(f"- Average wall time: {sum(linear_times)/len(linear_times):.2f} s\n\n")

        # Nonlinear summary
        nonlinear_times = []
        for profile_data in nonlinear_results.values():
            for case_data in [profile_data.get('max_N_case'), profile_data.get('min_dt_case')]:
                if case_data and isinstance(case_data.get('wall_time'), (int, float)):
                    nonlinear_times.append(case_data['wall_time'])

        if nonlinear_times:
            f.write(f"**Nonlinear solver:**\n")
            f.write(f"- Total runs: {len(nonlinear_times)}\n")
            f.write(f"- Max wall time: {max(nonlinear_times):.2f} s\n")
            f.write(f"- Min wall time: {min(nonlinear_times):.2f} s\n")
            f.write(f"- Average wall time: {sum(nonlinear_times)/len(nonlinear_times):.2f} s\n\n")

        f.write("---\n")
        f.write(f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")


def main():
    print("="*80)
    print("Maximum Wall Time Measurement Script")
    print("="*80)

    # Paths to datasets
    linear_dataset = 'dataset/hasegawa_mima_linear/successful/tasks.json'
    nonlinear_dataset = 'dataset/hasegawa_mima_nonlinear/successful/tasks.json'

    # Load tasks
    print("\n📂 Loading datasets...")
    linear_tasks = load_successful_tasks(linear_dataset)
    nonlinear_tasks = load_successful_tasks(nonlinear_dataset)
    print(f"  Linear: {len(linear_tasks)} successful tasks")
    print(f"  Nonlinear: {len(nonlinear_tasks)} successful tasks")

    # Find max complexity cases
    print("\n🔍 Finding maximum complexity cases per profile...")
    linear_profiles = find_max_complexity_per_profile(linear_tasks, 'linear')
    nonlinear_profiles = find_max_complexity_per_profile(nonlinear_tasks, 'nonlinear')

    print(f"\n  Linear profiles found: {list(linear_profiles.keys())}")
    print(f"  Nonlinear profiles found: {list(nonlinear_profiles.keys())}")

    # Display what we'll run
    print("\n📋 Cases to run:")
    print("\nLinear:")
    for profile, data in sorted(linear_profiles.items()):
        if data['max_N_case']:
            case = data['max_N_case']
            print(f"  {profile} - Max N: N={case['N']}, dt={case['dt']}, cg_atol={case['cg_atol']}")
        if data['min_dt_case']:
            case = data['min_dt_case']
            print(f"  {profile} - Min dt: N={case['N']}, dt={case['dt']}, cg_atol={case['cg_atol']}")

    print("\nNonlinear:")
    for profile, data in sorted(nonlinear_profiles.items()):
        if data['max_N_case']:
            case = data['max_N_case']
            print(f"  {profile} - Max N: N={case['N']}, dt={case['dt']}")
        if data['min_dt_case']:
            case = data['min_dt_case']
            print(f"  {profile} - Min dt: N={case['N']}, dt={case['dt']}")

    # Run simulations and measure wall time
    print("\n⏱️  Running simulations and measuring wall time...")

    # Linear simulations
    print("\n" + "="*80)
    print("LINEAR SOLVER SIMULATIONS")
    print("="*80)

    linear_results = {}
    for profile in sorted(linear_profiles.keys()):
        linear_results[profile] = {}

        # Max N case
        if linear_profiles[profile]['max_N_case']:
            case = linear_profiles[profile]['max_N_case']
            result = run_simulation_with_timing(
                'linear', profile, case['N'], case['dt'], case['cg_atol']
            )
            linear_results[profile]['max_N_case'] = result

        # Min dt case (only if different from max N case)
        if linear_profiles[profile]['min_dt_case']:
            case = linear_profiles[profile]['min_dt_case']
            # Check if this is different from max_N_case
            max_n_case = linear_profiles[profile]['max_N_case']
            if not max_n_case or (case['N'] != max_n_case['N'] or case['dt'] != max_n_case['dt']):
                result = run_simulation_with_timing(
                    'linear', profile, case['N'], case['dt'], case['cg_atol']
                )
                linear_results[profile]['min_dt_case'] = result
            else:
                # Reuse max_N_case result
                linear_results[profile]['min_dt_case'] = linear_results[profile]['max_N_case']

    # Nonlinear simulations
    print("\n" + "="*80)
    print("NONLINEAR SOLVER SIMULATIONS")
    print("="*80)

    nonlinear_results = {}
    for profile in sorted(nonlinear_profiles.keys()):
        nonlinear_results[profile] = {}

        # Max N case
        if nonlinear_profiles[profile]['max_N_case']:
            case = nonlinear_profiles[profile]['max_N_case']
            result = run_simulation_with_timing(
                'nonlinear', profile, case['N'], case['dt']
            )
            nonlinear_results[profile]['max_N_case'] = result

        # Min dt case (only if different from max N case)
        if nonlinear_profiles[profile]['min_dt_case']:
            case = nonlinear_profiles[profile]['min_dt_case']
            # Check if this is different from max_N_case
            max_n_case = nonlinear_profiles[profile]['max_N_case']
            if not max_n_case or (case['N'] != max_n_case['N'] or case['dt'] != max_n_case['dt']):
                result = run_simulation_with_timing(
                    'nonlinear', profile, case['N'], case['dt']
                )
                nonlinear_results[profile]['min_dt_case'] = result
            else:
                # Reuse max_N_case result
                nonlinear_results[profile]['min_dt_case'] = nonlinear_results[profile]['max_N_case']

    # Generate markdown report
    output_path = 'max_walltime_report.md'
    print(f"\n📝 Generating markdown report: {output_path}")
    generate_markdown_report(linear_results, nonlinear_results, output_path)

    print(f"\n✅ Report generated: {output_path}")
    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()
