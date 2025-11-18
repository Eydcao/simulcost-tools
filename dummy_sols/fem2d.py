import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.fem2d import get_fem2d_data, compare_energies_fem2d


def find_convergent_dx(profile, dx, cfl, energy_tolerance, var_threshold, multiplication_factor, max_iteration_num):
    """Iteratively decrease dx (grid resolution) until convergence is achieved with fixed other parameters.

    Args:
        profile: Profile name (p1, p2, p3)
        dx: Initial grid resolution
        cfl: CFL number for time step calculation
        energy_tolerance: Tolerance for energy comparison
        var_threshold: Threshold for energy conservation (coefficient of variation)
        multiplication_factor: Factor to multiply dx by each iteration
        max_iteration_num: Maximum number of iterations

    Returns:
        tuple: (converged, best_dx, cost_history, param_history)
    """
    dx_history = []
    cost_history = []
    param_history = []

    current_dx = dx
    converged = False
    best_dx = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with dx = {current_dx}, cfl = {cfl}")

        # Run simulation and load results (max_wall_time=-1 reads from config)
        _, cost_i = get_fem2d_data(profile=profile, dx=current_dx, cfl=cfl, max_wall_time=-1)
        cost_history.append(cost_i)
        dx_history.append(current_dx)
        param_history.append({"dx": current_dx, "cfl": cfl})

        # If we have previous results to compare with
        if len(dx_history) > 1:
            prev_dx = dx_history[-2]

            # Compare with previous results
            # - prev_dx (coarser): proposal with wall time constraint
            # - current_dx (finer): reference without constraint (ground truth)
            is_converged, metrics1, metrics2, avg_energy_diff = compare_energies_fem2d(
                profile1=profile,
                dx1=prev_dx,
                cfl1=cfl,
                profile2=profile,
                dx2=current_dx,
                cfl2=cfl,
                energy_tolerance=energy_tolerance,
                var_threshold=var_threshold,
                max_wall_time1=-1,  # Coarse/proposal: read from config
                max_wall_time2=None,  # Fine/reference: unconstrained
            )

            if is_converged:
                print(f"Convergence achieved between dx {prev_dx} and {current_dx}")
                best_dx = dx_history[-1]  # The finer grid that converged
                converged = True
                break
            else:
                print(f"No convergence between dx {prev_dx} and {current_dx}")

        # Prepare next dx using multiplication factor
        next_dx = current_dx * multiplication_factor
        current_dx = next_dx

    if converged:
        print(f"\nConvergent dx found: {best_dx}")
    else:
        print(f"\nNo convergence achieved within {max_iteration_num} iterations")

    return converged, best_dx, cost_history, param_history


def find_convergent_cfl(profile, dx, cfl, energy_tolerance, var_threshold, multiplication_factor, max_iteration_num):
    """Iteratively reduce cfl (CFL number) until convergence is achieved.

    Args:
        profile: Profile name (p1, p2, p3)
        dx: Grid resolution
        cfl: Initial CFL number
        energy_tolerance: Tolerance for energy comparison
        var_threshold: Threshold for energy conservation (coefficient of variation)
        multiplication_factor: Factor to multiply cfl by each iteration (should be < 1 to reduce cfl)
        max_iteration_num: Maximum number of iterations

    Returns:
        tuple: (converged, best_cfl, cost_history, param_history)
    """
    cfl_history = []
    cost_history = []
    param_history = []

    current_cfl = cfl
    converged = False
    best_cfl = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with dx = {dx}, cfl = {current_cfl}")

        # Run simulation and load results (max_wall_time=-1 reads from config)
        _, cost_i = get_fem2d_data(profile=profile, dx=dx, cfl=current_cfl, max_wall_time=-1)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)
        param_history.append({"dx": dx, "cfl": current_cfl})

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            # - prev_cfl (larger, less stable): proposal with wall time constraint
            # - current_cfl (smaller, more stable): reference without constraint (ground truth)
            is_converged, metrics1, metrics2, avg_energy_diff = compare_energies_fem2d(
                profile1=profile,
                dx1=dx,
                cfl1=prev_cfl,
                profile2=profile,
                dx2=dx,
                cfl2=current_cfl,
                energy_tolerance=energy_tolerance,
                var_threshold=var_threshold,
                max_wall_time1=-1,  # Less stable/proposal: read from config
                max_wall_time2=None,  # More stable/reference: unconstrained
            )

            if is_converged:
                print(f"Convergence achieved between cfl {prev_cfl} and {current_cfl}")
                best_cfl = cfl_history[-1]  # The smaller cfl that converged
                converged = True
                break
            else:
                print(f"No convergence between cfl {prev_cfl} and {current_cfl}")

        # Prepare next cfl using multiplication factor
        next_cfl = current_cfl * multiplication_factor
        current_cfl = next_cfl

    if converged:
        print(f"\nConvergent cfl found: {best_cfl}")
    else:
        print(f"\nNo convergence achieved within {max_iteration_num} iterations")

    return converged, best_cfl, cost_history, param_history
