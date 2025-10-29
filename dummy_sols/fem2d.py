import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.fem2d import get_fem2d_data, compare_energies_fem2d


def find_convergent_nx(
    profile, nx, cfl, newton_v_res_tol, energy_tolerance, var_threshold, multiplication_factor, max_iteration_num
):
    """Iteratively increase nx (grid resolution) until convergence is achieved with fixed other parameters.

    Args:
        profile: Profile name (p1, p2, p3)
        nx: Initial grid resolution
        cfl: CFL number for time step calculation
        newton_v_res_tol: Newton velocity residual tolerance
        energy_tolerance: Tolerance for energy comparison
        var_threshold: Threshold for energy conservation (coefficient of variation)
        multiplication_factor: Factor to multiply nx by each iteration
        max_iteration_num: Maximum number of iterations

    Returns:
        tuple: (converged, best_nx, cost_history, param_history)
    """
    nx_history = []
    cost_history = []
    param_history = []

    current_nx = nx
    converged = False
    best_nx = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with nx = {current_nx}, cfl = {cfl}, newton_v_res_tol = {newton_v_res_tol}")

        # Run simulation and load results
        _, cost_i = get_fem2d_data(profile=profile, nx=current_nx, cfl=cfl, newton_v_res_tol=newton_v_res_tol)
        cost_history.append(cost_i)
        nx_history.append(current_nx)
        param_history.append({"nx": current_nx, "cfl": cfl, "newton_v_res_tol": newton_v_res_tol})

        # If we have previous results to compare with
        if len(nx_history) > 1:
            prev_nx = nx_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, avg_energy_diff = compare_energies_fem2d(
                profile1=profile,
                nx1=prev_nx,
                cfl1=cfl,
                newton_v_res_tol1=newton_v_res_tol,
                profile2=profile,
                nx2=current_nx,
                cfl2=cfl,
                newton_v_res_tol2=newton_v_res_tol,
                energy_tolerance=energy_tolerance,
                var_threshold=var_threshold,
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


def find_convergent_cfl(
    profile, nx, cfl, newton_v_res_tol, energy_tolerance, var_threshold, multiplication_factor, max_iteration_num
):
    """Iteratively reduce cfl (CFL number) until convergence is achieved.

    Args:
        profile: Profile name (p1, p2, p3)
        nx: Grid resolution
        cfl: Initial CFL number
        newton_v_res_tol: Newton velocity residual tolerance
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
        print(f"\nRunning simulation with nx = {nx}, cfl = {current_cfl}, newton_v_res_tol = {newton_v_res_tol}")

        # Run simulation and load results
        _, cost_i = get_fem2d_data(profile=profile, nx=nx, cfl=current_cfl, newton_v_res_tol=newton_v_res_tol)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)
        param_history.append({"nx": nx, "cfl": current_cfl, "newton_v_res_tol": newton_v_res_tol})

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, avg_energy_diff = compare_energies_fem2d(
                profile1=profile,
                nx1=nx,
                cfl1=prev_cfl,
                newton_v_res_tol1=newton_v_res_tol,
                profile2=profile,
                nx2=nx,
                cfl2=current_cfl,
                newton_v_res_tol2=newton_v_res_tol,
                energy_tolerance=energy_tolerance,
                var_threshold=var_threshold,
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


def find_convergent_newton_v_res_tol(
    profile, nx, cfl, newton_v_res_tol, energy_tolerance, var_threshold, multiplication_factor, max_iteration_num
):
    """Iteratively reduce newton_v_res_tol (Newton convergence tolerance) until convergence is achieved.

    Args:
        profile: Profile name (p1, p2, p3)
        nx: Grid resolution
        cfl: CFL number
        newton_v_res_tol: Initial Newton velocity residual tolerance
        energy_tolerance: Tolerance for energy comparison
        var_threshold: Threshold for energy conservation (coefficient of variation)
        multiplication_factor: Factor to multiply newton_v_res_tol by each iteration (should be < 1 to reduce tolerance)
        max_iteration_num: Maximum number of iterations

    Returns:
        tuple: (converged, best_newton_v_res_tol, cost_history, param_history)
    """
    newton_v_res_tol_history = []
    cost_history = []
    param_history = []

    current_newton_v_res_tol = newton_v_res_tol
    converged = False
    best_newton_v_res_tol = None

    for i in range(max_iteration_num):
        # Round to avoid floating point issues
        current_newton_v_res_tol = round(current_newton_v_res_tol, 6)
        print(f"\nRunning simulation with nx = {nx}, cfl = {cfl}, newton_v_res_tol = {current_newton_v_res_tol}")

        # Run simulation and load results
        _, cost_i = get_fem2d_data(profile=profile, nx=nx, cfl=cfl, newton_v_res_tol=current_newton_v_res_tol)
        cost_history.append(cost_i)
        newton_v_res_tol_history.append(current_newton_v_res_tol)
        param_history.append({"nx": nx, "cfl": cfl, "newton_v_res_tol": current_newton_v_res_tol})

        # If we have previous results to compare with
        if len(newton_v_res_tol_history) > 1:
            prev_newton_v_res_tol = newton_v_res_tol_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, avg_energy_diff = compare_energies_fem2d(
                profile1=profile,
                nx1=nx,
                cfl1=cfl,
                newton_v_res_tol1=prev_newton_v_res_tol,
                profile2=profile,
                nx2=nx,
                cfl2=cfl,
                newton_v_res_tol2=current_newton_v_res_tol,
                energy_tolerance=energy_tolerance,
                var_threshold=var_threshold,
            )

            if is_converged:
                print(
                    f"Convergence achieved between newton_v_res_tol {prev_newton_v_res_tol} and {current_newton_v_res_tol}"
                )
                best_newton_v_res_tol = newton_v_res_tol_history[-1]  # The smaller tolerance that converged
                converged = True
                break
            else:
                print(f"No convergence between newton_v_res_tol {prev_newton_v_res_tol} and {current_newton_v_res_tol}")

        # Prepare next newton_v_res_tol using multiplication factor
        next_newton_v_res_tol = current_newton_v_res_tol * multiplication_factor
        current_newton_v_res_tol = next_newton_v_res_tol

    if converged:
        print(f"\nConvergent newton_v_res_tol found: {best_newton_v_res_tol}")
    else:
        print(f"\nNo convergence achieved within {max_iteration_num} iterations")

    return converged, best_newton_v_res_tol, cost_history, param_history
