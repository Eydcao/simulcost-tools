import argparse
import numpy as np
import sys
import os

# Import wrapper functions directly
from wrappers.euler_2d import get_res_euler_2d, compare_res_euler_2d


def find_convergent_n_grid_x(
    profile,
    n_grid_x,
    cfl,
    cg_tolerance,
    tolerance_rmse,
    multiplication_factor,
    max_iteration_num,
):
    """Iteratively increase n_grid_x (decrease dx) until convergence is achieved.

    Args:
        profile: Profile identifier (e.g., 'p1', 'p2', 'p3', 'p4')
        n_grid_x: Initial grid resolution in x-direction
        cfl: CFL number for timestep stability
        cg_tolerance: CG solver convergence tolerance
        tolerance_rmse: RMSE tolerance for convergence
        multiplication_factor: Factor to multiply n_grid_x by each iteration (e.g., 2.0)
        max_iteration_num: Maximum number of iterations

    Returns:
        converged: Boolean indicating if convergence was achieved
        best_n_grid_x: The grid resolution that achieved convergence
        cost_history: List of costs for each n_grid_x tried
        param_history: List of parameter dictionaries for each iteration
    """
    param_history = []
    cost_history = []
    converged = False
    best_n_grid_x = None

    for i in range(max_iteration_num):
        print(f"\nIteration {i+1}: Running simulation with n_grid_x = {n_grid_x}")

        # Run simulation
        _, cost_i = get_res_euler_2d(
            profile,
            n_grid_x,
            cfl=cfl,
            cg_tolerance=cg_tolerance,
        )

        cost_history.append(cost_i)
        param_history.append(
            {
                "n_grid_x": n_grid_x,
                "cfl": cfl,
                "cg_tolerance": cg_tolerance,
            }
        )

        # Check convergence if we have at least two results
        if len(param_history) > 1:
            prev_n_grid_x = param_history[-2]["n_grid_x"]

            is_converged, rmse = compare_res_euler_2d(
                profile,
                prev_n_grid_x,
                profile,
                n_grid_x,
                tolerance_rmse,
                cfl1=cfl,
                cg_tolerance1=cg_tolerance,
                cfl2=cfl,
                cg_tolerance2=cg_tolerance,
            )

            if is_converged:
                print(f"Convergence achieved between n_grid_x {prev_n_grid_x} and {n_grid_x}")
                best_n_grid_x = param_history[-1]["n_grid_x"]  # The finer of the two resolutions
                converged = True
                break
            else:
                print(f"No convergence between n_grid_x {prev_n_grid_x} and {n_grid_x}, RMSE = {rmse}")

        # Refine grid for next iteration
        n_grid_x = int(n_grid_x * multiplication_factor)

    if not converged and param_history:
        best_n_grid_x = param_history[-1]["n_grid_x"]

    return bool(converged), best_n_grid_x, cost_history, param_history


def find_convergent_cfl(
    profile,
    n_grid_x,
    cfl,
    cg_tolerance,
    tolerance_rmse,
    multiplication_factor,
    max_iteration_num,
):
    """Iteratively decrease cfl until convergence is achieved.

    Args:
        profile: Profile identifier (e.g., 'p1', 'p2', 'p3', 'p4')
        n_grid_x: Grid resolution in x-direction (fixed)
        cfl: Initial CFL number
        cg_tolerance: CG solver convergence tolerance (fixed)
        tolerance_rmse: RMSE tolerance for convergence
        multiplication_factor: Factor to multiply cfl by each iteration (e.g., 0.5)
        max_iteration_num: Maximum number of iterations

    Returns:
        converged: Boolean indicating if convergence was achieved
        best_cfl: The CFL that achieved convergence
        cost_history: List of costs for each cfl tried
        param_history: List of parameter dictionaries for each iteration
    """
    param_history = []
    cost_history = []
    converged = False
    best_cfl = None

    for i in range(max_iteration_num):
        print(f"\nIteration {i+1}: Running simulation with cfl = {cfl}")

        # Run simulation
        _, cost_i = get_res_euler_2d(
            profile,
            n_grid_x,
            cfl=cfl,
            cg_tolerance=cg_tolerance,
        )

        cost_history.append(cost_i)
        param_history.append(
            {
                "n_grid_x": n_grid_x,
                "cfl": cfl,
                "cg_tolerance": cg_tolerance,
            }
        )

        # Check convergence if we have at least two results
        if len(param_history) > 1:
            prev_cfl = param_history[-2]["cfl"]

            is_converged, rmse = compare_res_euler_2d(
                profile,
                n_grid_x,
                profile,
                n_grid_x,
                tolerance_rmse,
                cfl1=prev_cfl,
                cg_tolerance1=cg_tolerance,
                cfl2=cfl,
                cg_tolerance2=cg_tolerance,
            )

            if is_converged:
                print(f"Convergence achieved between cfl {prev_cfl} and {cfl}")
                best_cfl = param_history[-1]["cfl"]  # The smaller CFL that converged
                converged = True
                break
            else:
                print(f"No convergence between cfl {prev_cfl} and {cfl}, RMSE = {rmse}")

        # Reduce CFL for next iteration
        cfl = cfl * multiplication_factor

    if not converged and param_history:
        best_cfl = param_history[-1]["cfl"]

    return bool(converged), best_cfl, cost_history, param_history


def find_convergent_cg_tol(
    profile,
    n_grid_x,
    cfl,
    cg_tolerance,
    tolerance_rmse,
    multiplication_factor,
    max_iteration_num,
):
    """Iteratively decrease cg_tolerance until convergence is achieved.

    Args:
        profile: Profile identifier (e.g., 'p1', 'p2', 'p3', 'p4')
        n_grid_x: Grid resolution in x-direction (fixed)
        cfl: CFL number (fixed)
        cg_tolerance: Initial CG solver convergence tolerance
        tolerance_rmse: RMSE tolerance for convergence
        multiplication_factor: Factor to multiply cg_tolerance by each iteration (e.g., 0.1)
        max_iteration_num: Maximum number of iterations

    Returns:
        converged: Boolean indicating if convergence was achieved
        best_cg_tolerance: The cg_tolerance that achieved convergence
        cost_history: List of costs for each cg_tolerance tried
        param_history: List of parameter dictionaries for each iteration
    """
    param_history = []
    cost_history = []
    converged = False
    best_cg_tolerance = None

    for i in range(max_iteration_num):
        print(f"\nIteration {i+1}: Running simulation with cg_tolerance = {cg_tolerance}")

        # Run simulation
        _, cost_i = get_res_euler_2d(
            profile,
            n_grid_x,
            cfl=cfl,
            cg_tolerance=cg_tolerance,
        )

        cost_history.append(cost_i)
        param_history.append(
            {
                "n_grid_x": n_grid_x,
                "cfl": cfl,
                "cg_tolerance": cg_tolerance,
            }
        )

        # Check convergence with previous value if available
        if len(param_history) > 1:
            prev_cg_tolerance = param_history[-2]["cg_tolerance"]

            is_converged, rmse = compare_res_euler_2d(
                profile,
                n_grid_x,
                profile,
                n_grid_x,
                tolerance_rmse,
                cfl1=cfl,
                cg_tolerance1=prev_cg_tolerance,
                cfl2=cfl,
                cg_tolerance2=cg_tolerance,
            )

            if is_converged:
                print(f"Convergence achieved between cg_tolerance {prev_cg_tolerance} and {cg_tolerance}")
                best_cg_tolerance = param_history[-1]["cg_tolerance"]  # Current value that converged
                converged = True
                break
            else:
                print(f"No convergence between cg_tolerance {prev_cg_tolerance} and {cg_tolerance}, RMSE = {rmse}")

        # Reduce CG tolerance for next iteration
        cg_tolerance = cg_tolerance * multiplication_factor

    if not converged and param_history:
        best_cg_tolerance = param_history[-1]["cg_tolerance"]

    return bool(converged), best_cg_tolerance, cost_history, param_history


if __name__ == "__main__":
    ps = ["p1", "p2", "p3", "p4", "p5"]
    for p in ps:
        print(find_convergent_n_grid_x(p, 32, 0.25, 1e-6, 0.01, 2, 4))
        print(find_convergent_cfl(p, 128, 1.0, 1e-6, 0.01, 0.5, 5))
        print(find_convergent_cg_tol(p, 128, 0.25, 1e-2, 0.01, 0.1, 5))
