import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.unstruct_mpm import run_sim_unstruct_mpm, compare_energies_unstruct_mpm


def find_convergent_nx(profile, nx, n_part, cfl, energy_tolerance, var_threshold, multiplication_factor, max_iteration_num, case="cantilever"):
    """Iteratively increase nx (grid resolution) until convergence is achieved with fixed other parameters.
    Note: radii is fixed at 1.5 for all simulations.
    """
    nx_history = []
    cost_history = []
    param_history = []

    current_nx = nx
    converged = False
    best_nx = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with nx = {current_nx}, n_part = {n_part}, cfl = {cfl}")

        # Run simulation and load results
        cost_i, is_converged = run_sim_unstruct_mpm(profile=profile, nx=current_nx, n_part=n_part, cfl=cfl, case=case)
        cost_history.append(cost_i)
        nx_history.append(current_nx)
        param_history.append({"nx": current_nx, "n_part": n_part, "cfl": cfl})

        # If we have previous results to compare with
        if len(nx_history) > 1:
            prev_nx = nx_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, max_energy_diff = compare_energies_unstruct_mpm(
                profile1=profile,
                nx1=prev_nx,
                n_part1=n_part,
                cfl1=cfl,
                profile2=profile,
                nx2=current_nx,
                n_part2=n_part,
                cfl2=cfl,
                case1=case,
                case2=case,
                energy_tolerance=energy_tolerance,
                var_threshold=var_threshold
            )

            if is_converged:
                print(f"Convergence achieved between nx {prev_nx} and {current_nx}")
                best_nx = nx_history[-1]  # The finer grid that converged
                converged = True
                break
            else:
                print(f"No convergence between nx {prev_nx} and {current_nx}")

        # Prepare next nx using multiplication factor
        next_nx = int(current_nx * multiplication_factor)
        current_nx = next_nx

    if converged:
        print(f"\nConvergent nx found: {best_nx}")
    else:
        print(f"\nNo convergence achieved within {max_iteration_num} iterations")

    return converged, best_nx, cost_history, param_history


def find_convergent_n_part(profile, nx, n_part, cfl, energy_tolerance, var_threshold, multiplication_factor, max_iteration_num, case="cantilever"):
    """Iteratively increase n_part until convergence is achieved with fixed other parameters.
    Note: radii is fixed at 1.5 for all simulations.
    """
    n_part_history = []
    cost_history = []
    param_history = []

    current_n_part = n_part
    converged = False
    best_n_part = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with nx = {nx}, n_part = {current_n_part}, cfl = {cfl}")

        # Run simulation and load results
        cost_i, is_converged = run_sim_unstruct_mpm(profile=profile, nx=nx, n_part=current_n_part, cfl=cfl, case=case)
        cost_history.append(cost_i)
        n_part_history.append(current_n_part)
        param_history.append({"nx": nx, "n_part": current_n_part, "cfl": cfl})

        # If we have previous results to compare with
        if len(n_part_history) > 1:
            prev_n_part = n_part_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, max_energy_diff = compare_energies_unstruct_mpm(
                profile1=profile,
                nx1=nx,
                n_part1=prev_n_part,
                cfl1=cfl,
                profile2=profile,
                nx2=nx,
                n_part2=current_n_part,
                cfl2=cfl,
                case1=case,
                case2=case,
                energy_tolerance=energy_tolerance,
                var_threshold=var_threshold
            )

            if is_converged:
                print(f"Convergence achieved between n_part {prev_n_part} and {current_n_part}")
                best_n_part = n_part_history[-1]  # The higher particle count that converged
                converged = True
                break
            else:
                print(f"No convergence between n_part {prev_n_part} and {current_n_part}")

        # Prepare next n_part using multiplication factor
        next_n_part = int(current_n_part * multiplication_factor)
        current_n_part = next_n_part

    if converged:
        print(f"\nConvergent n_part found: {best_n_part}")
    else:
        print(f"\nNo convergence achieved within {max_iteration_num} iterations")

    return converged, best_n_part, cost_history, param_history


def find_convergent_cfl(profile, nx, n_part, cfl, energy_tolerance, var_threshold, multiplication_factor, max_iteration_num, case="cantilever"):
    """Iteratively reduce CFL number until convergence is achieved.
    Note: radii is fixed at 1.5 for all simulations.
    """
    cfl_history = []
    cost_history = []
    param_history = []

    current_cfl = cfl
    converged = False
    best_cfl = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with nx = {nx}, n_part = {n_part}, cfl = {current_cfl}")

        # Run simulation and load results
        cost_i, is_converged = run_sim_unstruct_mpm(profile=profile, nx=nx, n_part=n_part, cfl=current_cfl, case=case)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)
        param_history.append({"nx": nx, "n_part": n_part, "cfl": current_cfl})

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, max_energy_diff = compare_energies_unstruct_mpm(
                profile1=profile,
                nx1=nx,
                n_part1=n_part,
                cfl1=prev_cfl,
                profile2=profile,
                nx2=nx,
                n_part2=n_part,
                cfl2=current_cfl,
                case1=case,
                case2=case,
                energy_tolerance=energy_tolerance,
                var_threshold=var_threshold
            )

            if is_converged:
                print(f"Convergence achieved between cfl {prev_cfl} and {current_cfl}")
                best_cfl = cfl_history[-1]  # The smaller CFL that converged
                converged = True
                break
            else:
                print(f"No convergence between cfl {prev_cfl} and {current_cfl}")

        # Prepare next CFL using multiplication factor
        next_cfl = current_cfl * multiplication_factor
        current_cfl = next_cfl

    if converged:
        print(f"\nConvergent CFL found: {best_cfl}")
    else:
        print(f"\nNo convergence achieved within {max_iteration_num} iterations")

    return converged, best_cfl, cost_history, param_history


if __name__ == "__main__":
    # Example usage
    profile = "p1"
    # Profile-specific initial nx values
    initial_nx_values = {
        "p1": 11,   # cantilever: Lx=11.0
        "p2": 35,   # vibration_bar: Lx=35.0
        "p3": 10    # disk_collision: Lx=1.0
    }
    initial_nx = initial_nx_values[profile]
    n_part = 2
    cfl = 0.001
    energy_tolerance = 1e-6
    var_threshold = 0.01
    # Test nx convergence
    converged, best_nx, cost_history, param_history = find_convergent_nx(
        profile=profile,
        nx=initial_nx,  # Use profile-specific initial value
        n_part=n_part,
        cfl=cfl,
        energy_tolerance=energy_tolerance,
        var_threshold=var_threshold,
        multiplication_factor=2,
        max_iteration_num=4,
        case="cantilever"
    )
    
    print(f"NX convergence test: {converged}, best_nx: {best_nx}")
