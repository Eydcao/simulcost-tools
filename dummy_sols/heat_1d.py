import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrappers import *


def find_convergent_cfl(profile, initial_cfl, initial_n_space, tolerance, max_iter, multiplication_factor):
    """Iteratively reduce CFL number until convergence is achieved."""
    cfl_history = []
    cost_history = []
    param_history = []

    # Fix the n_space for the CFL searching
    n_space = initial_n_space

    current_cfl = initial_cfl
    converged = False
    best_cfl = None

    for i in range(max_iter):
        print(f"\nRunning simulation with CFL = {current_cfl}")

        # Run simulation and load results
        cost_i = run_sim_heat_1d(profile, current_cfl, n_space)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)
        param_history.append({"cfl": current_cfl, "n_space": n_space})

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            is_converged, _ = compare_res_heat_1d(profile, prev_cfl, n_space, profile, current_cfl, n_space, tolerance)

            if is_converged:
                print(f"Convergence achieved between CFL {prev_cfl} and {current_cfl}")
                best_cfl = cfl_history[-1]  # The larger of the two CFLs that converged
                converged = True
                break
            else:
                print(f"No convergence between CFL {prev_cfl} and {current_cfl}")

        # Prepare next CFL using multiplication factor
        current_cfl = current_cfl * multiplication_factor

    if not converged and len(cfl_history) > 1:
        # Check if last two simulations converged
        is_converged, _ = compare_res_heat_1d(
            profile, cfl_history[-2], n_space, profile, cfl_history[-1], n_space, tolerance
        )
        if is_converged:
            best_cfl = cfl_history[-2]
            converged = True

    if converged:
        print(f"\nConvergent CFL number found: {best_cfl}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(cfl_history) > 1:
            best_cfl = cfl_history[-1]
            print(f"Smallest tested CFL: {best_cfl}")
        else:
            best_cfl = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(is_converged), best_cfl, cost_history, param_history


def find_convergent_n_space(profile, initial_n_space, cfl, tolerance, max_iter, multiplication_factor):
    """Iteratively increase n_space number until convergence is achieved."""
    n_space_history = []
    cost_history = []
    param_history = []

    # Fix the CFL for the n_space searching
    current_cfl = cfl

    current_n_space = initial_n_space
    converged = False
    best_n_space = None

    for i in range(max_iter):
        print(f"\nRunning simulation with n_space = {current_n_space}")

        # Run simulation and load results
        cost_i = run_sim_heat_1d(profile, current_cfl, current_n_space)
        cost_history.append(cost_i)
        n_space_history.append(current_n_space)
        param_history.append({"cfl": current_cfl, "n_space": current_n_space})

        # If we have previous results to compare with
        if len(n_space_history) > 1:
            prev_n_space = n_space_history[-2]

            # Compare with previous results (with interpolation if needed)
            is_converged, _ = compare_res_heat_1d(
                profile, current_cfl, prev_n_space, profile, current_cfl, current_n_space, tolerance
            )

            if is_converged:
                print(f"Convergence achieved between n_space {prev_n_space} and {current_n_space}")
                best_n_space = n_space_history[-1]  # The coarser of the two that converged
                converged = True
                break
            else:
                print(f"No convergence between n_space {prev_n_space} and {current_n_space}")

        # Prepare next n_space using multiplication factor
        next_n_space = current_n_space * multiplication_factor
        current_n_space = next_n_space

    if not converged and len(n_space_history) > 1:
        # Check if last two simulations converged
        is_converged, _ = compare_res_heat_1d(
            profile, current_cfl, n_space_history[-2], profile, current_cfl, n_space_history[-1], tolerance
        )
        if is_converged:
            best_n_space = n_space_history[-2]
            converged = True

    if converged:
        print(f"\nConvergent n_space found: {best_n_space}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(n_space_history) > 1:
            best_n_space = n_space_history[-1]
            print(f"Finest tested n_space: {best_n_space}")
        else:
            best_n_space = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(is_converged), best_n_space, cost_history, param_history
