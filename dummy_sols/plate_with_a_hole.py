import argparse
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.plate_with_a_hole import run_sim_plate_with_a_hole, get_error_metric
import json


def find_convergent_nx(profile, nx, ny, tolerance_error, individual_error_tolerance, multiplication_factor, max_iteration_num):
    """Iteratively increase nx until convergence is achieved with fixed ny."""
    nx_history = []
    cost_history = []
    param_history = []

    current_nx = nx
    converged = False
    best_nx = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with nx = {current_nx}, ny = {ny}")

        # Run simulation and load results
        cost_i = run_sim_plate_with_a_hole(profile=profile, nx=current_nx, ny=ny)
        cost_history.append(cost_i)
        nx_history.append(current_nx)
        param_history.append({"nx": current_nx, "ny": ny})

        # If we have previous results to compare with
        if len(nx_history) > 1:
            prev_nx = nx_history[-2]

            # Get error metrics from meta.json for both simulations
            sim_dir_prev = f"sim_res/plate_with_a_hole/{profile}_nx_{prev_nx}_ny_{ny}"
            sim_dir_curr = f"sim_res/plate_with_a_hole/{profile}_nx_{current_nx}_ny_{ny}"
            
            try:
                # Read error from meta.json
                with open(os.path.join(sim_dir_prev, "meta.json"), "r") as f:
                    meta_prev = json.load(f)
                    error_prev = meta_prev.get("error", None)
                
                with open(os.path.join(sim_dir_curr, "meta.json"), "r") as f:
                    meta_curr = json.load(f)
                    error_curr = meta_curr.get("error", None)

                if error_prev is not None and error_curr is not None:
                    # Check if error improvement is within tolerance
                    error_differnece = abs(error_prev - error_curr)
                    
                    if error_differnece < tolerance_error and error_curr < individual_error_tolerance:
                        print(f"Convergence achieved between nx {prev_nx} and {current_nx}")
                        print(f"Individual error tolerance met: error={error_curr:.2e} <= tolerance={individual_error_tolerance:.2e}")
                        best_nx = nx_history[-1]  # The finer grid that converged
                        converged = True
                        break
                    else:
                        print(f"No convergence between nx {prev_nx} and {current_nx} (error difference: {error_differnece:.2e}, error: {error_curr:.2e})")
                else:
                    print(f"Could not extract error metrics for comparison")
            except (FileNotFoundError, KeyError) as e:
                print(f"Could not read error metrics: {e}")

        # Prepare next nx using multiplication factor
        next_nx = int(current_nx * multiplication_factor)
        current_nx = next_nx

    if converged:
        print(f"\nConvergent nx found: {best_nx}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(nx_history) > 1:
            best_nx = nx_history[-1]
            print(f"Finest tested nx: {best_nx}")
        else:
            best_nx = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_nx, cost_history, param_history


def find_convergent_ny(profile, nx, ny, tolerance_error, individual_error_tolerance, multiplication_factor, max_iteration_num):
    """Iteratively increase ny until convergence is achieved with fixed nx."""
    ny_history = []
    cost_history = []
    param_history = []

    current_ny = ny
    converged = False
    best_ny = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with nx = {nx}, ny = {current_ny}")

        # Run simulation and load results
        cost_i = run_sim_plate_with_a_hole(profile=profile, nx=nx, ny=current_ny)
        cost_history.append(cost_i)
        ny_history.append(current_ny)
        param_history.append({"nx": nx, "ny": current_ny})

        # If we have previous results to compare with
        if len(ny_history) > 1:
            prev_ny = ny_history[-2]

            # Get error metrics from meta.json for both simulations
            sim_dir_prev = f"sim_res/plate_with_a_hole/{profile}_nx_{nx}_ny_{prev_ny}"
            sim_dir_curr = f"sim_res/plate_with_a_hole/{profile}_nx_{nx}_ny_{current_ny}"
            
            try:
                # Read error from meta.json
                with open(os.path.join(sim_dir_prev, "meta.json"), "r") as f:
                    meta_prev = json.load(f)
                    error_prev = meta_prev.get("error", None)
                
                with open(os.path.join(sim_dir_curr, "meta.json"), "r") as f:
                    meta_curr = json.load(f)
                    error_curr = meta_curr.get("error", None)

                if error_prev is not None and error_curr is not None:                    
                    # Check if error improvement is within tolerance
                    error_differnece = abs(error_prev - error_curr)
                    
                    if error_differnece < tolerance_error and error_curr < individual_error_tolerance:
                        print(f"Convergence achieved between ny {prev_ny} and {current_ny}")
                        print(f"Individual error tolerance met: error={error_curr:.2e} <= tolerance={individual_error_tolerance:.2e}")
                        best_ny = ny_history[-1]  # The finer grid that converged
                        converged = True
                        break
                    else:
                        print(f"No convergence between ny {prev_ny} and {current_ny} (error difference: {error_differnece:.2e}, error: {error_curr:.2e})")
                else:
                    print(f"Could not extract error metrics for comparison")
            except (FileNotFoundError, KeyError) as e:
                print(f"Could not read error metrics: {e}")

        # Prepare next ny using multiplication factor
        next_ny = int(current_ny * multiplication_factor)
        current_ny = next_ny

    if converged:
        print(f"\nConvergent ny found: {best_ny}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(ny_history) > 1:
            best_ny = ny_history[-1]
            print(f"Finest tested ny: {best_ny}")
        else:
            best_ny = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_ny, cost_history, param_history


def find_optimal_nx(
    profile,
    ny,
    nx,
    tolerance_error,
    individual_error_tolerance,
    search_range_min,
    search_range_max,
    search_range_slice_num,
    multiplication_factor,
    max_iteration_num,
):
    """
    Grid search over nx ∈ [search_range_min, search_range_max] for optimal mesh resolution.
    For each nx, iterate ny until spatial convergence is achieved.

    Returns
    -------
    is_converged_optimal : bool
        Whether optimal_nx achieved spatial convergence.
    optimal_param : (int | None, int | None)
        (optimal_nx, optimal_ny). None if no convergent solution found.
    optimal_cost_history : list[float] | None
        Cost history for optimal_nx corresponding to converged ny sequence.
        None if no convergent solution found.
    param_history : list
        Full parameter exploration history.
    """
    nx_values = np.linspace(search_range_min, search_range_max, search_range_slice_num, dtype=int)
    param_history = []
    nx_results = []  # Save key info for each nx (when converged)

    for nx in nx_values:
        nx = int(nx)
        print(f"\n=== Testing nx = {nx} ===")

        is_converged, best_ny, cost_history, one_param_history = find_convergent_ny(
            profile, nx, ny, tolerance_error, individual_error_tolerance, multiplication_factor, max_iteration_num
        )

        # Record ny exploration trajectory for each nx
        param_history.append(one_param_history)

        # If convergent ny found, save to results pool
        if best_ny is not None:
            total_cost = sum(cost_history)
            nx_results.append(
                {
                    "nx": nx,
                    "best_ny": best_ny,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
                }
            )
            print(f"nx = {nx}: Best ny = {best_ny}, Total Cost = {total_cost}")
        else:
            print(f"nx = {nx}: No convergent ny found")

    # Select convergent solution with minimum total cost
    if nx_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in nx_results]))
        opt_rec = nx_results[min_cost_idx]

        optimal_nx = opt_rec["nx"]
        optimal_ny = opt_rec["best_ny"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal nx found: {optimal_nx} with ny = {optimal_ny}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_nx = optimal_ny = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal nx found")

    optimal_param = (optimal_nx, optimal_ny)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


def find_optimal_ny(
    profile,
    nx,
    ny,
    tolerance_error,
    individual_error_tolerance,
    search_range_min,
    search_range_max,
    search_range_slice_num,
    multiplication_factor,
    max_iteration_num,
):
    """
    Grid search over ny ∈ [search_range_min, search_range_max] for optimal mesh resolution.
    For each ny, iterate nx until spatial convergence is achieved.

    Returns
    -------
    is_converged_optimal : bool
        Whether optimal_ny achieved spatial convergence.
    optimal_param : (int | None, int | None)
        (optimal_nx, optimal_ny). None if no convergent solution found.
    optimal_cost_history : list[float] | None
        Cost history for optimal_ny corresponding to converged nx sequence.
        None if no convergent solution found.
    param_history : list
        Full parameter exploration history.
    """
    ny_values = np.linspace(search_range_min, search_range_max, search_range_slice_num, dtype=int)
    param_history = []
    ny_results = []  # Save key info for each ny (when converged)

    for ny in ny_values:
        ny = int(ny)
        print(f"\n=== Testing ny = {ny} ===")

        is_converged, best_nx, cost_history, one_param_history = find_convergent_nx(
            profile, nx, ny, tolerance_error, individual_error_tolerance, multiplication_factor, max_iteration_num
        )

        # Record nx exploration trajectory for each ny
        param_history.append(one_param_history)

        # If convergent nx found, save to results pool
        if best_nx is not None:
            total_cost = sum(cost_history)
            ny_results.append(
                {
                    "ny": ny,
                    "best_nx": best_nx,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
                }
            )
            print(f"ny = {ny}: Best nx = {best_nx}, Total Cost = {total_cost}")
        else:
            print(f"ny = {ny}: No convergent nx found")

    # Select convergent solution with minimum total cost
    if ny_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in ny_results]))
        opt_rec = ny_results[min_cost_idx]

        optimal_ny = opt_rec["ny"]
        optimal_nx = opt_rec["best_nx"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal ny found: {optimal_ny} with nx = {optimal_nx}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_ny = optimal_nx = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal ny found")

    optimal_param = (optimal_nx, optimal_ny)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal parameters for Plate with Hole simulation")

    # Search mode selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["nx", "ny"],
        required=True,
        help="Choose which parameter to search: 'nx' or 'ny'",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile configuration")

    # Controllable parameters
    parser.add_argument("--nx", type=int, default=40, help="Initial number of elements in x direction")
    parser.add_argument("--ny", type=int, default=40, help="Initial number of elements in y direction")

    # Tolerance parameters
    parser.add_argument("--tolerance_error", type=float, default=0.01, help="Error tolerance for convergence checking")
    parser.add_argument("--individual_error_tolerance", type=float, default=0.1, help="Individual error tolerance for absolute error checking")

    # Search parameters for iterative tasks
    parser.add_argument(
        "--multiplication_factor",
        type=float,
        default=2.0,
        help="Factor to multiply/divide parameter values in iterative search",
    )
    parser.add_argument(
        "--max_iteration_num", type=int, default=5, help="Maximum number of iterations for iterative search"
    )

    # Search parameters for 0-shot tasks
    parser.add_argument(
        "--search_range_min", type=int, default=20, help="Minimum value for 0-shot parameter search range"
    )
    parser.add_argument(
        "--search_range_max", type=int, default=80, help="Maximum value for 0-shot parameter search range"
    )
    parser.add_argument(
        "--search_range_slice_num", type=int, default=7, help="Number of slices for 0-shot parameter search range"
    )

    args = parser.parse_args()

    if args.task == "nx":
        print("\n=== Starting nx convergence search ===")
        is_converged, best_nx, cost_history, param_history = find_convergent_nx(
            profile=args.profile,
            nx=args.nx,
            ny=args.ny,
            tolerance_error=args.tolerance_error,
            individual_error_tolerance=args.individual_error_tolerance,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_nx is not None:
            print(f"\nRecommended nx: {best_nx}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent nx found, total cost: {sum(cost_history)}")

    elif args.task == "ny":
        print("\n=== Starting ny convergence search ===")
        is_converged, best_ny, cost_history, param_history = find_convergent_ny(
            profile=args.profile,
            nx=args.nx,
            ny=args.ny,
            tolerance_error=args.tolerance_error,
            individual_error_tolerance=args.individual_error_tolerance,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_ny is not None:
            print(f"\nRecommended ny: {best_ny}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent ny found, total cost: {sum(cost_history)}")

    else:
        print(f"\nTask type '{args.task}' is not supported.")
