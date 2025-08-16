import argparse
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.heat_steady_2d import run_sim_heat_steady_2d, compare_res_heat_steady_2d


def find_convergent_dx(
    profile, dx, relax, error_threshold, t_init, tolerance_rmse, multiplication_factor, max_iteration_num
):
    """Iteratively reduce dx (increase resolution) until convergence is achieved."""
    dx_history = []
    cost_history = []
    param_history = []

    current_dx = dx
    converged = False
    best_dx = None

    for i in range(max_iteration_num):
        print(
            f"\nRunning simulation with dx = {current_dx}, relax = {relax}, error_threshold = {error_threshold}, T_init = {t_init}"
        )

        # Run simulation and load results
        cost_i = run_sim_heat_steady_2d(profile, current_dx, relax, error_threshold, t_init)
        cost_history.append(cost_i)
        dx_history.append(current_dx)
        param_history.append({"dx": current_dx, "relax": relax, "error_threshold": error_threshold, "t_init": t_init})

        # If we have previous results to compare with
        if len(dx_history) > 1:
            prev_dx = dx_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, rmse = compare_res_heat_steady_2d(
                profile,
                prev_dx,
                relax,
                error_threshold,
                t_init,
                profile,
                current_dx,
                relax,
                error_threshold,
                t_init,
                tolerance_rmse,
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
        print("\nMaximum iterations reached without convergence")
        if len(dx_history) > 1:
            best_dx = dx_history[-1]
            print(f"Finest tested dx: {best_dx}")
        else:
            best_dx = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_dx, cost_history, param_history


def find_optimal_relax(
    profile,
    dx,
    error_threshold,
    t_init,
    tolerance_rmse,
    search_range_min,
    search_range_max,
    search_range_slice_num,
    multiplication_factor,
    max_iteration_num,
):
    """
    Grid search over relax ∈ [search_range_min, search_range_max] for optimal relaxation parameter.
    For each relax, iterate dx until spatial convergence is achieved.

    Returns
    -------
    is_converged_optimal : bool
        Whether optimal_relax achieved spatial convergence.
    optimal_param : (float | None, float | None)
        (optimal_relax, optimal_dx). None if no convergent solution found.
    optimal_cost_history : list[float] | None
        Cost history for optimal_relax corresponding to converged dx sequence.
        None if no convergent solution found.
    param_history : list
        Full parameter exploration history.
    """
    relax_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
    param_history = []
    relax_results = []  # Save key info for each relax (when converged)

    for relax in relax_values:
        relax = round(float(relax), 2)
        print(f"\n=== Testing relax = {relax} ===")

        is_converged, best_dx, cost_history, one_param_history = find_convergent_dx(
            profile, dx, relax, error_threshold, t_init, tolerance_rmse, multiplication_factor, max_iteration_num
        )

        # Record dx exploration trajectory for each relax
        param_history.append(one_param_history)

        # If convergent dx found, save to results pool
        if best_dx is not None:
            total_cost = sum(cost_history)
            relax_results.append(
                {
                    "relax": relax,
                    "best_dx": best_dx,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
                }
            )
            print(f"relax = {relax}: Best dx = {best_dx}, Total Cost = {total_cost}")
        else:
            print(f"relax = {relax}: No convergent dx found")

    # Select convergent solution with minimum total cost
    if relax_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in relax_results]))
        opt_rec = relax_results[min_cost_idx]

        optimal_relax = opt_rec["relax"]
        optimal_dx = opt_rec["best_dx"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal relax found: {optimal_relax} with dx = {optimal_dx}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_relax = optimal_dx = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal relax found")

    optimal_param = (optimal_relax, optimal_dx)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


def find_optimal_t_init(
    profile,
    dx,
    relax,
    error_threshold,
    tolerance_rmse,
    search_range_min,
    search_range_max,
    search_range_slice_num,
    multiplication_factor,
    max_iteration_num,
):
    """
    Grid search over t_init ∈ [search_range_min, search_range_max] for optimal initial temperature.
    For each t_init, iterate dx until spatial convergence is achieved.

    Returns
    -------
    is_converged_optimal : bool
        Whether optimal_t_init achieved spatial convergence.
    optimal_param : (float | None, float | None)
        (optimal_t_init, optimal_dx). None if no convergent solution found.
    optimal_cost_history : list[float] | None
        Cost history for optimal_t_init corresponding to converged dx sequence.
        None if no convergent solution found.
    param_history : list
        Full parameter exploration history.
    """
    t_init_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
    param_history = []
    t_init_results = []  # Save key info for each t_init (when converged)

    for t_init in t_init_values:
        t_init = round(float(t_init), 2)
        print(f"\n=== Testing t_init = {t_init} ===")

        is_converged, best_dx, cost_history, one_param_history = find_convergent_dx(
            profile, dx, relax, error_threshold, t_init, tolerance_rmse, multiplication_factor, max_iteration_num
        )

        # Record dx exploration trajectory for each t_init
        param_history.append(one_param_history)

        # If convergent dx found, save to results pool
        if best_dx is not None:
            total_cost = sum(cost_history)
            t_init_results.append(
                {
                    "t_init": t_init,
                    "best_dx": best_dx,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
                }
            )
            print(f"t_init = {t_init}: Best dx = {best_dx}, Total Cost = {total_cost}")
        else:
            print(f"t_init = {t_init}: No convergent dx found")

    # Select convergent solution with minimum total cost
    if t_init_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in t_init_results]))
        opt_rec = t_init_results[min_cost_idx]

        optimal_t_init = opt_rec["t_init"]
        optimal_dx = opt_rec["best_dx"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal t_init found: {optimal_t_init} with dx = {optimal_dx}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_t_init = optimal_dx = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal t_init found")

    optimal_param = (optimal_t_init, optimal_dx)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


def find_convergent_error_threshold(
    profile, dx, relax, t_init, error_threshold, tolerance_rmse, multiplication_factor, max_iteration_num
):
    """Iteratively reduce error_threshold until convergence is achieved."""
    error_threshold_history = []
    cost_history = []
    param_history = []

    current_error_threshold = error_threshold
    converged = False
    best_error_threshold = None

    for i in range(max_iteration_num):
        print(
            f"\nRunning simulation with error_threshold = {current_error_threshold}, dx = {dx}, relax = {relax}, T_init = {t_init}"
        )

        # Run simulation and load results
        cost_i = run_sim_heat_steady_2d(profile, dx, relax, current_error_threshold, t_init)
        cost_history.append(cost_i)
        error_threshold_history.append(current_error_threshold)
        param_history.append({"error_threshold": current_error_threshold, "dx": dx, "relax": relax, "t_init": t_init})

        # If we have previous results to compare with
        if len(error_threshold_history) > 1:
            prev_error_threshold = error_threshold_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, rmse = compare_res_heat_steady_2d(
                profile,
                dx,
                relax,
                prev_error_threshold,
                t_init,
                profile,
                dx,
                relax,
                current_error_threshold,
                t_init,
                tolerance_rmse,
            )

            if is_converged:
                print(
                    f"Convergence achieved between error_threshold {prev_error_threshold} and {current_error_threshold}"
                )
                best_error_threshold = error_threshold_history[-1]  # The stricter threshold that converged
                converged = True
                break
            else:
                print(f"No convergence between error_threshold {prev_error_threshold} and {current_error_threshold}")

        # Prepare next error_threshold using multiplication factor
        next_error_threshold = current_error_threshold * multiplication_factor
        current_error_threshold = next_error_threshold

    if converged:
        print(f"\nConvergent error_threshold found: {best_error_threshold}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(error_threshold_history) > 1:
            best_error_threshold = error_threshold_history[-1]
            print(f"Strictest tested error_threshold: {best_error_threshold}")
        else:
            best_error_threshold = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_error_threshold, cost_history, param_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal parameters for Heat Steady 2D simulation")

    # Search mode selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["dx", "relax", "t_init", "error_threshold"],
        required=True,
        help="Choose which parameter to search: 'dx', 'relax', 't_init', or 'error_threshold'",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile configuration")

    # Controllable parameters
    parser.add_argument("--dx", type=float, default=0.01, help="Initial grid spacing to start testing")
    parser.add_argument("--relax", type=float, default=1.0, help="SOR relaxation parameter")
    parser.add_argument("--error_threshold", type=float, default=1e-8, help="Convergence threshold for iterations")
    parser.add_argument("--t_init", type=float, default=0.0, help="Initial temperature field value")

    # Tolerance parameters
    parser.add_argument("--tolerance_rmse", type=float, default=1e-6, help="RMSE tolerance for convergence checking")

    # Search parameters for iterative tasks
    parser.add_argument(
        "--multiplication_factor",
        type=float,
        default=0.5,
        help="Factor to multiply parameter values in iterative search",
    )
    parser.add_argument(
        "--max_iteration_num", type=int, default=7, help="Maximum number of iterations for iterative search"
    )

    # Search parameters for 0-shot tasks
    parser.add_argument(
        "--search_range_min", type=float, default=0.1, help="Minimum value for 0-shot parameter search range"
    )
    parser.add_argument(
        "--search_range_max", type=float, default=1.9, help="Maximum value for 0-shot parameter search range"
    )
    parser.add_argument(
        "--search_range_slice_num", type=int, default=10, help="Number of slices for 0-shot parameter search range"
    )

    args = parser.parse_args()

    if args.task == "dx":
        print("\n=== Starting dx convergence search ===")
        is_converged, best_dx, cost_history, param_history = find_convergent_dx(
            profile=args.profile,
            dx=args.dx,
            relax=args.relax,
            error_threshold=args.error_threshold,
            t_init=args.t_init,
            tolerance_rmse=args.tolerance_rmse,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_dx is not None:
            print(f"\nRecommended dx: {best_dx}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent dx found, total cost: {sum(cost_history)}")

    elif args.task == "relax":
        print("\n=== Starting relax parameter search ===")
        is_converged, optimal_param, optimal_cost_history, param_history = find_optimal_relax(
            profile=args.profile,
            dx=args.dx,
            error_threshold=args.error_threshold,
            t_init=args.t_init,
            tolerance_rmse=args.tolerance_rmse,
            search_range_min=args.search_range_min,
            search_range_max=args.search_range_max,
            search_range_slice_num=args.search_range_slice_num,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        optimal_relax, optimal_dx = optimal_param
        if optimal_relax is not None:
            print(f"\nRecommended relax: {optimal_relax} with dx: {optimal_dx}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal relax found")

    elif args.task == "t_init":
        print("\n=== Starting t_init parameter search ===")
        is_converged, optimal_param, optimal_cost_history, param_history = find_optimal_t_init(
            profile=args.profile,
            dx=args.dx,
            relax=args.relax,
            error_threshold=args.error_threshold,
            tolerance_rmse=args.tolerance_rmse,
            search_range_min=args.search_range_min,
            search_range_max=args.search_range_max,
            search_range_slice_num=args.search_range_slice_num,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        optimal_t_init, optimal_dx = optimal_param
        if optimal_t_init is not None:
            print(f"\nRecommended t_init: {optimal_t_init} with dx: {optimal_dx}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal t_init found")

    elif args.task == "error_threshold":
        print("\n=== Starting error_threshold convergence search ===")
        is_converged, best_error_threshold, cost_history, param_history = find_convergent_error_threshold(
            profile=args.profile,
            dx=args.dx,
            relax=args.relax,
            t_init=args.t_init,
            error_threshold=args.error_threshold,
            tolerance_rmse=args.tolerance_rmse,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_error_threshold is not None:
            print(f"\nRecommended error_threshold: {best_error_threshold}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent error_threshold found, total cost: {sum(cost_history)}")

    else:
        print(f"\nTask type '{args.task}' is not supported.")
