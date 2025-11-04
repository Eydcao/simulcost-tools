#!/usr/bin/env python3
"""Measure wall time for max complexity cases in hasegawa_mima datasets."""

import json
import time
import subprocess
import sys
from collections import defaultdict

PYTHON_PATH = "/home/ubuntu/miniconda3/envs/casebench/bin/python"


def find_max_complexity_cases(tasks, solver_type):
    """Find max N and min dt cases for each profile."""
    profile_max_N = defaultdict(lambda: {'N': 0, 'dt': None, 'cg_atol': None})
    profile_min_dt = defaultdict(lambda: {'N': None, 'dt': float('inf'), 'cg_atol': None})

    for task in tasks:
        profile = task['profile']
        target_param = task['target_parameter']
        optimal_value = task['results']['optimal_parameter_value']
        non_target = task['non_target_parameters']

        if target_param == 'N':
            N = optimal_value
            dt = non_target.get('dt')
            cg_atol = non_target.get('cg_atol') if solver_type == 'linear' else None
        elif target_param == 'dt':
            N = non_target.get('N')
            dt = optimal_value
            cg_atol = non_target.get('cg_atol') if solver_type == 'linear' else None
        else:  # cg_atol (linear only)
            N = non_target.get('N')
            dt = non_target.get('dt')
            cg_atol = optimal_value

        # Track max N
        if N is not None and N > profile_max_N[profile]['N']:
            profile_max_N[profile] = {'N': N, 'dt': dt, 'cg_atol': cg_atol}

        # Track min dt
        if dt is not None and dt < profile_min_dt[profile]['dt']:
            profile_min_dt[profile] = {'N': N, 'dt': dt, 'cg_atol': cg_atol}

    return dict(profile_max_N), dict(profile_min_dt)


def run_simulation(solver_type, profile, N, dt, cg_atol=None):
    """Run simulation and measure wall time."""
    if solver_type == 'linear':
        runner = 'runners/hasegawa_mima_linear.py'
        if cg_atol is None:
            cg_atol = 1e-8
        cmd = [
            PYTHON_PATH, runner,
            f'--config-name={profile}',
            f'N={N}', f'dt={dt}', f'cg_atol={cg_atol}',
            'analytical=false', 'verbose=false'
        ]
    else:
        runner = 'runners/hasegawa_mima_nonlinear.py'
        cmd = [
            PYTHON_PATH, runner,
            f'--config-name={profile}',
            f'N={N}', f'dt={dt}',
            'verbose=false'
        ]

    print(f"\nRunning {solver_type} - {profile}: N={N}, dt={dt}", end='', flush=True)
    if cg_atol is not None:
        print(f", cg_atol={cg_atol}", end='', flush=True)
    print(" ...", flush=True)

    start_time = time.time()
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        wall_time = time.time() - start_time
        print(f"  ✅ Completed in {wall_time:.2f}s", flush=True)
        return {'N': N, 'dt': dt, 'cg_atol': cg_atol, 'wall_time': wall_time, 'success': True}
    except subprocess.CalledProcessError as e:
        wall_time = time.time() - start_time
        print(f"  ❌ Failed after {wall_time:.2f}s: {e}", flush=True)
        return {'N': N, 'dt': dt, 'cg_atol': cg_atol, 'wall_time': wall_time, 'success': False}


def generate_markdown(linear_results, nonlinear_results):
    """Generate markdown report."""
    with open('max_walltime_report.md', 'w') as f:
        f.write("# Maximum Wall Time Measurements\n\n")
        f.write("Wall time measurements for the most computationally expensive cases.\n\n")

        # Linear
        f.write("## Hasegawa-Mima Linear\n\n")
        f.write("| Profile | Case Type | N | dt | cg_atol | Wall Time (s) | Status |\n")
        f.write("|---------|-----------|---|----|---------|--------------:|--------|\n")

        for profile in sorted(linear_results.keys()):
            for case_type, case_label in [('max_N', 'Max N'), ('min_dt', 'Min dt')]:
                result = linear_results[profile].get(case_type)
                if result:
                    status = '✅' if result['success'] else '❌'
                    cg_str = f"{result['cg_atol']:.1e}" if result['cg_atol'] is not None else 'N/A'
                    f.write(f"| {profile} | {case_label} | {result['N']} | {result['dt']} | "
                           f"{cg_str} | {result['wall_time']:.2f} | {status} |\n")

        # Nonlinear
        f.write("\n## Hasegawa-Mima Nonlinear\n\n")
        f.write("| Profile | Case Type | N | dt | Wall Time (s) | Status |\n")
        f.write("|---------|-----------|---|----|--------------:|--------|\n")

        for profile in sorted(nonlinear_results.keys()):
            for case_type, case_label in [('max_N', 'Max N'), ('min_dt', 'Min dt')]:
                result = nonlinear_results[profile].get(case_type)
                if result:
                    status = '✅' if result['success'] else '❌'
                    f.write(f"| {profile} | {case_label} | {result['N']} | {result['dt']} | "
                           f"{result['wall_time']:.2f} | {status} |\n")

        # Summary
        f.write("\n## Summary\n\n")

        linear_times = [r[ct]['wall_time'] for r in linear_results.values()
                       for ct in ['max_N', 'min_dt'] if ct in r and r[ct]['success']]
        nonlinear_times = [r[ct]['wall_time'] for r in nonlinear_results.values()
                          for ct in ['max_N', 'min_dt'] if ct in r and r[ct]['success']]

        if linear_times:
            f.write(f"**Linear solver:**\n")
            f.write(f"- Runs: {len(linear_times)}\n")
            f.write(f"- Max: {max(linear_times):.2f}s, Min: {min(linear_times):.2f}s, "
                   f"Avg: {sum(linear_times)/len(linear_times):.2f}s\n\n")

        if nonlinear_times:
            f.write(f"**Nonlinear solver:**\n")
            f.write(f"- Runs: {len(nonlinear_times)}\n")
            f.write(f"- Max: {max(nonlinear_times):.2f}s, Min: {min(nonlinear_times):.2f}s, "
                   f"Avg: {sum(nonlinear_times)/len(nonlinear_times):.2f}s\n\n")

        f.write(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")


def main():
    print("="*80, flush=True)
    print("Maximum Wall Time Measurement", flush=True)
    print("="*80, flush=True)

    # Load datasets
    print("\nLoading datasets...", flush=True)
    with open('dataset/hasegawa_mima_linear/successful/tasks.json', 'r') as f:
        linear_tasks = json.load(f)['tasks']
    with open('dataset/hasegawa_mima_nonlinear/successful/tasks.json', 'r') as f:
        nonlinear_tasks = json.load(f)['tasks']
    print(f"  Linear: {len(linear_tasks)} tasks", flush=True)
    print(f"  Nonlinear: {len(nonlinear_tasks)} tasks", flush=True)

    # Find max complexity cases
    print("\nFinding max complexity cases...", flush=True)
    linear_max_N, linear_min_dt = find_max_complexity_cases(linear_tasks, 'linear')
    nonlinear_max_N, nonlinear_min_dt = find_max_complexity_cases(nonlinear_tasks, 'nonlinear')

    # Run linear simulations
    print("\n" + "="*80, flush=True)
    print("LINEAR SIMULATIONS", flush=True)
    print("="*80, flush=True)

    linear_results = {}
    for profile in sorted(linear_max_N.keys()):
        linear_results[profile] = {}

        # Max N
        case = linear_max_N[profile]
        result = run_simulation('linear', profile, case['N'], case['dt'], case['cg_atol'])
        linear_results[profile]['max_N'] = result

        # Min dt (if different)
        min_case = linear_min_dt[profile]
        if min_case['N'] != case['N'] or min_case['dt'] != case['dt']:
            result = run_simulation('linear', profile, min_case['N'], min_case['dt'], min_case['cg_atol'])
            linear_results[profile]['min_dt'] = result
        else:
            linear_results[profile]['min_dt'] = result  # Reuse

    # Run nonlinear simulations
    print("\n" + "="*80, flush=True)
    print("NONLINEAR SIMULATIONS", flush=True)
    print("="*80, flush=True)

    nonlinear_results = {}
    for profile in sorted(nonlinear_max_N.keys()):
        nonlinear_results[profile] = {}

        # Max N
        case = nonlinear_max_N[profile]
        result = run_simulation('nonlinear', profile, case['N'], case['dt'])
        nonlinear_results[profile]['max_N'] = result

        # Min dt (if different)
        min_case = nonlinear_min_dt[profile]
        if min_case['N'] != case['N'] or min_case['dt'] != case['dt']:
            result = run_simulation('nonlinear', profile, min_case['N'], min_case['dt'])
            nonlinear_results[profile]['min_dt'] = result
        else:
            nonlinear_results[profile]['min_dt'] = result  # Reuse

    # Generate report
    print("\n" + "="*80, flush=True)
    print("Generating report...", flush=True)
    generate_markdown(linear_results, nonlinear_results)
    print("✅ Report saved: max_walltime_report.md", flush=True)
    print("="*80, flush=True)


if __name__ == '__main__':
    main()
