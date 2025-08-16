import argparse
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.heat_steady_2d import compare_res_heat_steady_2d, get_res_heat_steady_2d, compute_heat_steady_metrics


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

        # Run simulation and load results - use enhanced get function but extract cost only
        _, _, _, _, metadata = get_res_heat_steady_2d(profile, current_dx, relax, error_threshold, t_init)
        cost_i = metadata["cost"]
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
):
    relax_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
    param_history = []
    relax_results = []  # Save key info for each relax (when converged)

    for relax in relax_values:
        relax = round(float(relax), 2)
        print(f"\n=== Testing relax = {relax} ===")
        print(f"Using fixed parameters: dx = {dx}, error_threshold = {error_threshold}, t_init = {t_init}")

        # Get simulation results with validity checking
        T, X, Y, iter_count, metadata = get_res_heat_steady_2d(profile, dx, relax, error_threshold, t_init)
        cost = metadata["cost"]

        # Check solution validity
        metrics = compute_heat_steady_metrics(T, X, Y)
        is_valid = metrics["temperature_valid"]

        # Also check if solver detected numerical instability
        if metadata["numerical_instability"]:
            is_valid = False

        # Record parameter combination
        param_entry = {"relax": relax, "dx": dx, "error_threshold": error_threshold, "t_init": t_init}
        param_history.append(param_entry)

        # Only consider valid solutions
        if is_valid:
            total_cost = cost
            relax_results.append(
                {
                    "relax": relax,
                    "total_cost": total_cost,
                    "cost_history": [cost],
                }
            )
            print(f"relax = {relax}: VALID - Total Cost = {total_cost}")
        else:
            print(f"relax = {relax}: INVALID - Solution failed validity check")

    # Select solution with minimum total cost
    if relax_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in relax_results]))
        opt_rec = relax_results[min_cost_idx]

        optimal_relax = opt_rec["relax"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = True  # All runs are considered converged

        print(f"\nOptimal relax found: {optimal_relax}")
        print(f"Optimal cost: {sum(optimal_cost_history)}")
    else:
        optimal_relax = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal relax found")

    return (
        is_converged_optimal,
        optimal_relax,
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
):
    t_init_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
    param_history = []
    t_init_results = []  # Save key info for each t_init (when converged)

    for t_init in t_init_values:
        t_init = round(float(t_init), 2)
        print(f"\n=== Testing t_init = {t_init} ===")
        print(f"Using fixed parameters: dx = {dx}, relax = {relax}, error_threshold = {error_threshold}")

        # Get simulation results with validity checking
        T, X, Y, iter_count, metadata = get_res_heat_steady_2d(profile, dx, relax, error_threshold, t_init)
        cost = metadata["cost"]

        # Check solution validity
        metrics = compute_heat_steady_metrics(T, X, Y)
        is_valid = metrics["temperature_valid"]

        # Also check if solver detected numerical instability
        if metadata["numerical_instability"]:
            is_valid = False

        # Record parameter combination
        param_entry = {"t_init": t_init, "dx": dx, "relax": relax, "error_threshold": error_threshold}
        param_history.append(param_entry)

        # Only consider valid solutions
        if is_valid:
            total_cost = cost
            t_init_results.append(
                {
                    "t_init": t_init,
                    "total_cost": total_cost,
                    "cost_history": [cost],
                }
            )
            print(f"t_init = {t_init}: VALID - Total Cost = {total_cost}")
        else:
            print(f"t_init = {t_init}: INVALID - Solution failed validity check")

    # Select solution with minimum total cost
    if t_init_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in t_init_results]))
        opt_rec = t_init_results[min_cost_idx]

        optimal_t_init = opt_rec["t_init"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = True  # All runs are considered converged

        print(f"\nOptimal t_init found: {optimal_t_init}")
        print(f"Optimal cost: {sum(optimal_cost_history)}")
    else:
        optimal_t_init = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal t_init found")

    return (
        is_converged_optimal,
        optimal_t_init,
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

        # Run simulation and load results - use enhanced get function but extract cost only
        _, _, _, _, metadata = get_res_heat_steady_2d(profile, dx, relax, current_error_threshold, t_init)
        cost_i = metadata["cost"]
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
