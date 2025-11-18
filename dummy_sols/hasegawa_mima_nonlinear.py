import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.hasegawa_mima_nonlinear import get_results, compare_solutions


def find_convergent_N(profile, N, dt, tolerance_rmse, multiplication_factor, max_iteration_num):
    """
    Iteratively increase N (grid resolution) until convergence is achieved with FIXED dt.

    Goal: Find the best N for a given dt value.
    - Proposal runs: Use the target dt (may violate CFL for high N)
    - Reference runs: Use CFL-safe dt (scaled inversely with N: dt_ref = dt_target * N_prev / N_current)

    This function is responsible for scaling dt for the reference to maintain CFL stability.

    Note: error_history[i] represents the RMSE comparison between
    param_history[i] and param_history[i+1], so error_history is one element shorter.
    """
    N_history = []
    cost_history = []
    param_history = []
    error_history = []  # Note: Will be one element shorter than param_history

    current_N = N
    # dt stays FIXED throughout the search - we're finding best N for THIS dt
    fixed_dt = dt
    converged = False
    best_N = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with N = {current_N}, dt = {fixed_dt:.6e}")

        # Get simulation results with FIXED dt (proposal run with wall time constraint)
        cost_i, _, _ = get_results(profile=profile, N=current_N, dt=fixed_dt)
        cost_history.append(cost_i)
        N_history.append(current_N)
        param_history.append({"N": current_N, "dt": fixed_dt})

        # If this is not the first iteration, compare with previous resolution
        if i > 0:
            previous_N = N_history[-2]

            # Compare current resolution with previous resolution
            # Proposal: uses target dt (may violate CFL)
            # Reference: uses CFL-safe dt (scaled by N ratio)
            params1 = {"N": previous_N, "dt": fixed_dt}

            # Scale dt for reference to maintain CFL stability (dt ∝ 1/N)
            dt_ref = fixed_dt * previous_N / current_N
            params2 = {"N": current_N, "dt": dt_ref}

            print(f"   Comparing: proposal (N={previous_N}, dt={fixed_dt:.6e}) vs reference (N={current_N}, dt={dt_ref:.6e})")

            is_converged, cost1, cost2, rmse_diff = compare_solutions(profile, params1, params2, tolerance_rmse)
            error_history.append(rmse_diff)

            if is_converged:
                print(f"Convergence achieved with N = {current_N}, dt = {fixed_dt:.6e}, RMSE diff = {rmse_diff:.6e}")
                best_N = current_N
                converged = True
                break
            else:
                print(f"No convergence with N = {current_N}, dt = {fixed_dt:.6e}, RMSE diff = {rmse_diff:.6e}")

        # Prepare next N using multiplication factor
        # dt stays FIXED - only N changes
        next_N = int(current_N * multiplication_factor)
        current_N = next_N

    if converged:
        print(f"\nConvergent N found: {best_N} (for dt = {fixed_dt:.6e})")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(N_history) > 0:
            best_N = N_history[-1]
            print(f"Highest tested: N = {best_N} (for dt = {fixed_dt:.6e})")
        else:
            best_N = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_N, cost_history, param_history


def find_convergent_dt(profile, N, dt, tolerance_rmse, multiplication_factor, max_iteration_num):
    """
    Iteratively decrease dt (time step) until convergence is achieved.

    Note: error_history[i] represents the RMSE comparison between
    param_history[i] and param_history[i+1], so error_history is one element shorter.
    """
    dt_history = []
    cost_history = []
    param_history = []
    error_history = []  # Note: Will be one element shorter than param_history

    current_dt = dt
    converged = False
    best_dt = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with N = {N}, dt = {current_dt}")

        # Get simulation results
        cost_i, _, _ = get_results(profile=profile, N=N, dt=current_dt)
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

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_dt, cost_history, param_history
