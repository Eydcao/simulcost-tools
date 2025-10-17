import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only what we need to avoid circular dependencies
import subprocess
import json
from pathlib import Path


# Import wrapper functions directly to avoid duplication
from wrappers.euler_2d import run_sim_euler_2d, compare_res_euler_2d


def find_convergent_n_grid_x(
    profile,
    testcase,
    n_grid_x,
    cfl,
    cg_tolerance,
    start_frame,
    end_frame,
    tolerance_rmse,
    multiplication_factor,
    max_iteration_num,
):
    """Iteratively increase n_grid_x (decrease dx) until convergence is achieved.

    Args:
        profile: Profile identifier (e.g., 'p1', 'p2', 'p3', 'p4')
        testcase: Test case number (0-3)
        n_grid_x: Initial grid resolution in x-direction
        cfl: CFL number for timestep stability
        cg_tolerance: CG solver convergence tolerance
        start_frame: Starting frame
        end_frame: Ending frame
        tolerance_rmse: RMSE tolerance for convergence
        multiplication_factor: Factor to multiply n_grid_x by each iteration
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
        cost_i = run_sim_euler_2d(
            profile,
            testcase,
            n_grid_x,
            start_frame=start_frame,
            end_frame=end_frame,
            cfl=cfl,
            cg_tolerance=cg_tolerance,
        )

        cost_history.append(cost_i)
        param_history.append(
            {
                "n_grid_x": n_grid_x,
                "cfl": cfl,
                "cg_tolerance": cg_tolerance,
                "testcase": testcase,
            }
        )

        # Check convergence if we have at least two results
        if len(param_history) > 1:
            prev_n_grid_x = param_history[-2]["n_grid_x"]

            is_converged, _, _, rmse = compare_res_euler_2d(
                profile,
                testcase,
                prev_n_grid_x,
                profile,
                testcase,
                n_grid_x,
                tolerance_rmse,
                start_frame=start_frame,
                end_frame=end_frame,
                cfl1=cfl,
                cg_tolerance1=cg_tolerance,
                cfl2=cfl,
                cg_tolerance2=cg_tolerance,
            )

            if is_converged:
                print(f"Convergence achieved between n_grid_x {prev_n_grid_x} and {n_grid_x}")
                best_n_grid_x = param_history[-1]["n_grid_x"]  # The finer of the two resolutions that converged
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
    testcase,
    n_grid_x,
    cfl,
    cg_tolerance,
    start_frame,
    end_frame,
    tolerance_rmse,
    division_factor,
    max_iteration_num,
):
    """Iteratively decrease cfl until convergence is achieved.

    Args:
        profile: Profile identifier (e.g., 'p1', 'p2', 'p3', 'p4')
        testcase: Test case number (0-3)
        n_grid_x: Grid resolution in x-direction (fixed)
        cfl: Initial CFL number
        cg_tolerance: CG solver convergence tolerance (fixed)
        start_frame: Starting frame
        end_frame: Ending frame
        tolerance_rmse: RMSE tolerance for convergence
        division_factor: Factor to divide cfl by each iteration
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
        cost_i = run_sim_euler_2d(
            profile,
            testcase,
            n_grid_x,
            start_frame=start_frame,
            end_frame=end_frame,
            cfl=cfl,
            cg_tolerance=cg_tolerance,
        )

        cost_history.append(cost_i)
        param_history.append(
            {
                "n_grid_x": n_grid_x,
                "cfl": cfl,
                "cg_tolerance": cg_tolerance,
                "testcase": testcase,
            }
        )

        # Check convergence if we have at least two results
        if len(param_history) > 1:
            prev_cfl = param_history[-2]["cfl"]

            is_converged, _, _, rmse = compare_res_euler_2d(
                profile,
                testcase,
                n_grid_x,
                profile,
                testcase,
                n_grid_x,
                tolerance_rmse,
                start_frame=start_frame,
                end_frame=end_frame,
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
        cfl = cfl / division_factor

    if not converged and param_history:
        best_cfl = param_history[-1]["cfl"]

    return bool(converged), best_cfl, cost_history, param_history


def grid_search_cg_tolerance(
    profile,
    testcase,
    n_grid_x,
    cfl,
    cg_tolerance_values,
    start_frame,
    end_frame,
    tolerance_rmse,
):
    """Grid search over cg_tolerance values, selecting the one with minimum cost that maintains convergence.

    This is a 0-shot only parameter - we don't do iterative refinement, just compare consecutive
    values from the provided list.

    Args:
        profile: Profile identifier (e.g., 'p1', 'p2', 'p3', 'p4')
        testcase: Test case number (0-3)
        n_grid_x: Grid resolution in x-direction (fixed)
        cfl: CFL number (fixed)
        cg_tolerance_values: List of cg_tolerance values to try
        start_frame: Starting frame
        end_frame: Ending frame
        tolerance_rmse: RMSE tolerance for convergence check

    Returns:
        converged: Boolean indicating if any convergence was achieved
        best_cg_tolerance: The cg_tolerance with minimum cost among converged results
        cost_history: List of costs for each cg_tolerance tried
        param_history: List of parameter dictionaries for each iteration
    """
    param_history = []
    cost_history = []
    converged = False
    best_cg_tolerance = None

    for cg_tolerance in cg_tolerance_values:
        print(f"\nRunning simulation with cg_tolerance = {cg_tolerance}")

        # Run simulation
        cost_i = run_sim_euler_2d(
            profile,
            testcase,
            n_grid_x,
            start_frame=start_frame,
            end_frame=end_frame,
            cfl=cfl,
            cg_tolerance=cg_tolerance,
        )

        cost_history.append(cost_i)
        param_history.append(
            {
                "n_grid_x": n_grid_x,
                "cfl": cfl,
                "cg_tolerance": cg_tolerance,
                "testcase": testcase,
            }
        )

        # Check convergence with previous value if available
        if len(param_history) > 1:
            prev_cg_tolerance = param_history[-2]["cg_tolerance"]

            is_converged, _, _, rmse = compare_res_euler_2d(
                profile,
                testcase,
                n_grid_x,
                profile,
                testcase,
                n_grid_x,
                tolerance_rmse,
                start_frame=start_frame,
                end_frame=end_frame,
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

    if not converged and param_history:
        best_cg_tolerance = param_history[-1]["cg_tolerance"]

    return bool(converged), best_cg_tolerance, cost_history, param_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dummy solver for Euler 2D simulations")
    parser.add_argument(
        "--task",
        type=str,
        default="n_grid_x",
        choices=["n_grid_x", "cfl", "cg_tolerance"],
        help="Which parameter to optimize",
    )
    parser.add_argument("--profile", type=str, default="p1", help="Profile name (p1, p2, p3, p4)")
    parser.add_argument("--testcase", type=int, default=0, help="Test case (0, 1, 2, 3)")
    parser.add_argument("--n_grid_x", type=int, default=16, help="Initial grid resolution")
    parser.add_argument("--cfl", type=float, default=0.5, help="Initial CFL number")
    parser.add_argument("--cg_tolerance", type=float, default=1e-7, help="Initial CG tolerance")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame")
    parser.add_argument("--end_frame", type=int, default=20, help="End frame")
    parser.add_argument("--tolerance_rmse", type=float, default=0.05, help="RMSE tolerance for convergence")
    parser.add_argument("--max_iteration_num", type=int, default=5, help="Maximum iterations for iterative search")

    args = parser.parse_args()

    if args.task == "n_grid_x":
        print(f"\n{'='*60}")
        print(f"Finding convergent n_grid_x for profile {args.profile}")
        print(f"{'='*60}")
        converged, best_n_grid_x, cost_history, param_history = find_convergent_n_grid_x(
            args.profile,
            args.testcase,
            args.n_grid_x,
            args.cfl,
            args.cg_tolerance,
            args.start_frame,
            args.end_frame,
            args.tolerance_rmse,
            multiplication_factor=2.0,
            max_iteration_num=args.max_iteration_num,
        )
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"Converged: {converged}")
        print(f"Best n_grid_x: {best_n_grid_x}")
        print(f"Cost history: {cost_history}")
        print(f"{'='*60}")

    elif args.task == "cfl":
        print(f"\n{'='*60}")
        print(f"Finding convergent CFL for profile {args.profile}")
        print(f"{'='*60}")
        converged, best_cfl, cost_history, param_history = find_convergent_cfl(
            args.profile,
            args.testcase,
            args.n_grid_x,
            args.cfl,
            args.cg_tolerance,
            args.start_frame,
            args.end_frame,
            args.tolerance_rmse,
            division_factor=2.0,
            max_iteration_num=args.max_iteration_num,
        )
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"Converged: {converged}")
        print(f"Best CFL: {best_cfl}")
        print(f"Cost history: {cost_history}")
        print(f"{'='*60}")

    elif args.task == "cg_tolerance":
        print(f"\n{'='*60}")
        print(f"Grid search for optimal cg_tolerance for profile {args.profile}")
        print(f"{'='*60}")
        # Define candidate cg_tolerance values (from tight to loose)
        cg_tolerance_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        converged, best_cg_tolerance, cost_history, param_history = grid_search_cg_tolerance(
            args.profile,
            args.testcase,
            args.n_grid_x,
            args.cfl,
            cg_tolerance_values,
            args.start_frame,
            args.end_frame,
            args.tolerance_rmse,
        )
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"Converged: {converged}")
        print(f"Best cg_tolerance: {best_cg_tolerance}")
        print(f"Cost history: {cost_history}")
        print(f"{'='*60}")
