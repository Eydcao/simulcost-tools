import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.hasegawa_mima_nonlinear import get_results, compare_solutions


def find_convergent_N(profile, N, dt, tolerance_rmse, multiplication_factor, max_iteration_num):
    """Iteratively increase N (grid resolution) until convergence is achieved."""
    N_history = []
    cost_history = []
    param_history = []
    error_history = []

    current_N = N
    converged = False
    best_N = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with N = {current_N}, dt = {dt}")

        # Get simulation results
        cost_i, _ = get_results(profile=profile, N=current_N, dt=dt)
        cost_history.append(cost_i)
        N_history.append(current_N)
        param_history.append({"N": current_N, "dt": dt})

        # If this is not the first iteration, compare with previous resolution
        if i > 0:
            previous_N = N_history[-2]

            # Compare current resolution with previous resolution
            params1 = {"N": previous_N, "dt": dt}
            params2 = {"N": current_N, "dt": dt}

            is_converged, cost1, cost2, rmse_diff = compare_solutions(profile, params1, params2, tolerance_rmse)
            error_history.append(rmse_diff)

            if is_converged:
                print(f"Convergence achieved with N = {current_N}, RMSE diff = {rmse_diff:.6e}")
                best_N = current_N
                converged = True
                break
            else:
                print(f"No convergence with N = {current_N}, RMSE diff = {rmse_diff:.6e}")
        else:
            error_history.append(None)  # No comparison for first iteration

        # Prepare next N using multiplication factor
        next_N = int(current_N * multiplication_factor)
        current_N = next_N

    if converged:
        print(f"\nConvergent N found: {best_N}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(N_history) > 0:
            best_N = N_history[-1]
            print(f"Highest tested N: {best_N}")
        else:
            best_N = None

    # Return trajectory information
    trajectory = {
        "parameter_name": "N",
        "initial_value": N,
        "optimal_value": best_N,
        "converged": converged,
        "parameter_history": param_history,
        "cost_history": cost_history,
        "error_history": error_history,
        "N_history": N_history
    }

    return trajectory


def find_convergent_dt(profile, N, dt, tolerance_rmse, multiplication_factor, max_iteration_num):
    """Iteratively decrease dt (time step) until convergence is achieved."""
    dt_history = []
    cost_history = []
    param_history = []
    error_history = []

    current_dt = dt
    converged = False
    best_dt = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with N = {N}, dt = {current_dt}")

        # Get simulation results
        cost_i, _ = get_results(profile=profile, N=N, dt=current_dt)
        cost_history.append(cost_i)
        dt_history.append(current_dt)
        param_history.append({"N": N, "dt": current_dt})

        # If this is not the first iteration, compare with previous time step
        if i > 0:
            previous_dt = dt_history[-2]

            # Compare current time step with previous time step
            params1 = {"N": N, "dt": previous_dt}
            params2 = {"N": N, "dt": current_dt}

            is_converged, cost1, cost2, rmse_diff = compare_solutions(profile, params1, params2, tolerance_rmse)
            error_history.append(rmse_diff)

            if is_converged:
                print(f"Convergence achieved with dt = {current_dt}, RMSE diff = {rmse_diff:.6e}")
                best_dt = current_dt
                converged = True
                break
            else:
                print(f"No convergence with dt = {current_dt}, RMSE diff = {rmse_diff:.6e}")
        else:
            error_history.append(None)  # No comparison for first iteration

        # Prepare next dt using multiplication factor
        next_dt = current_dt * multiplication_factor
        current_dt = next_dt

    if converged:
        print(f"\nConvergent dt found: {best_dt}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(dt_history) > 0:
            best_dt = dt_history[-1]
            print(f"Smallest tested dt: {best_dt}")
        else:
            best_dt = None

    # Return trajectory information
    trajectory = {
        "parameter_name": "dt",
        "initial_value": dt,
        "optimal_value": best_dt,
        "converged": converged,
        "parameter_history": param_history,
        "cost_history": cost_history,
        "error_history": error_history,
        "dt_history": dt_history
    }

    return trajectory


def generate_dummy_solution(profile, target_param, initial_params, tolerance_rmse, multiplication_factor, max_iteration_num):
    """
    Generate a dummy solution trajectory for parameter optimization.

    Args:
        profile: Configuration profile name
        target_param: Parameter to optimize ("N" or "dt")
        initial_params: Dictionary with initial parameter values
        tolerance_rmse: RMSE tolerance for convergence
        multiplication_factor: Factor for parameter updates
        max_iteration_num: Maximum number of iterations

    Returns:
        trajectory: Dictionary with optimization trajectory
    """
    if target_param == "N":
        return find_convergent_N(
            profile=profile,
            N=initial_params["N"],
            dt=initial_params["dt"],
            tolerance_rmse=tolerance_rmse,
            multiplication_factor=multiplication_factor,
            max_iteration_num=max_iteration_num
        )
    elif target_param == "dt":
        return find_convergent_dt(
            profile=profile,
            N=initial_params["N"],
            dt=initial_params["dt"],
            tolerance_rmse=tolerance_rmse,
            multiplication_factor=multiplication_factor,
            max_iteration_num=max_iteration_num
        )
    else:
        raise ValueError(f"Unsupported target parameter: {target_param}")
