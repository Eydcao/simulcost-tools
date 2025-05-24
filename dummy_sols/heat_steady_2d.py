import argparse
import numpy as np

# Append abs path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrappers import *


def find_optimal_dx(profile, initial_dx, relax, error_threshold, T_init, tolerance, max_iter):
    """Iteratively halve dx until convergence is achieved."""
    param_history = []
    cost_history = []

    current_dx = initial_dx
    converged = False
    best_dx = None

    for i in range(max_iter):
        print(f"\nRunning simulation with dx = {current_dx}")

        # Run simulation and track cost
        cost_i, num_steps = run_sim_heat_steady_2d(profile, current_dx, relax, error_threshold, T_init)
        cost_history.append(cost_i)
        param_history.append({"dx": current_dx, "relax": relax, "T_init": T_init, "error_threshold": error_threshold})

        # If we have previous results to compare with
        if len(param_history) > 1:
            prev_dx = param_history[-2]["dx"]

            # Compare with previous results
            is_converged, _ = compare_res_heat_steady_2d(
                profile,
                prev_dx,
                relax,
                error_threshold,
                T_init,
                profile,
                current_dx,
                relax,
                error_threshold,
                T_init,
                tolerance,
            )

            if is_converged:
                print(f"Convergence achieved between dx {prev_dx} and {current_dx}")
                best_dx = prev_dx  # The coarser of the two grids that converged
                converged = True
                break
            else:
                print(f"No convergence between dx {prev_dx} and {current_dx}")

        # Halve the dx for next iteration
        next_dx = current_dx / 2
        current_dx = next_dx

    if converged:
        print(f"\nConvergent dx found: {best_dx}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(param_history) > 0:
            best_dx = param_history[-1]["dx"]
            print(f"Finest tested dx: {best_dx}")
        else:
            best_dx = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(is_converged), best_dx, cost_history, param_history


def grid_search_relax(profile, dx, relax_values, error_threshold, T_init, max_iter):
    """Perform grid search for optimal relaxation factor."""
    relax_costs = {}
    param_history = []

    for relax in relax_values:
        print(f"\nTesting relaxation factor: {relax}")
        cost, num_steps = run_sim_heat_steady_2d(profile, dx, relax, error_threshold, T_init)
        relax_costs[relax] = cost
        param_history.append({"dx": dx, "relax": relax, "T_init": T_init, "error_threshold": error_threshold})
        print(f"Cost for relax={relax}: {cost}")

    # Find relaxation factor with minimum cost
    best_relax = min(relax_costs, key=relax_costs.get)
    min_cost = relax_costs[best_relax]

    print(f"\nOptimal relaxation factor: {best_relax} with cost {min_cost}")
    print(f"All costs: {relax_costs}")

    return True, best_relax, relax_costs, param_history


def grid_search_T_init(profile, dx, relax, error_threshold, T_init_values, max_iter):
    """Perform grid search for optimal initial temperature."""
    T_init_costs = {}
    param_history = []

    for T_init in T_init_values:
        print(f"\nTesting initial temperature: {T_init}")
        cost, num_steps = run_sim_heat_steady_2d(profile, dx, relax, error_threshold, T_init)
        T_init_costs[T_init] = cost
        param_history.append({"dx": dx, "relax": relax, "T_init": T_init, "error_threshold": error_threshold})
        print(f"Cost for T_init={T_init}: {cost}")

    # Find initial temperature with minimum cost
    best_T_init = min(T_init_costs, key=T_init_costs.get)
    min_cost = T_init_costs[best_T_init]

    print(f"\nOptimal initial temperature: {best_T_init} with cost {min_cost}")
    print(f"All costs: {T_init_costs}")

    return True, best_T_init, T_init_costs, param_history


def find_optimal_error_threshold(profile, dx, relax, T_init, initial_error, tolerance, max_iter):
    """Decrease error threshold by factors of 10 until convergence."""
    param_history = []
    cost_history = []

    current_error = initial_error
    best_error = None
    converged = False

    for i in range(max_iter):
        print(f"\nRunning simulation with error threshold = {current_error}")

        # Run simulation and track cost
        cost_i, num_steps = run_sim_heat_steady_2d(profile, dx, relax, current_error, T_init)
        cost_history.append(cost_i)
        param_history.append({"dx": dx, "relax": relax, "T_init": T_init, "error_threshold": current_error})

        # If we have previous results to compare with
        if len(param_history) > 1:
            prev_error = param_history[-2]["error_threshold"]

            # Compare with previous results
            is_converged, _ = compare_res_heat_steady_2d(
                profile, dx, relax, prev_error, T_init, profile, dx, relax, current_error, T_init, tolerance
            )

            if is_converged:
                print(f"Convergence achieved between error {prev_error} and {current_error}")
                best_error = prev_error  # The looser error threshold that still converged
                converged = True
                break
            else:
                print(f"No convergence between error {prev_error} and {current_error}")

        # Decrease error threshold by factor of 10
        next_error = current_error / 10
        current_error = next_error

    if converged:
        print(f"\nConvergent error threshold found: {best_error}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(param_history) > 0:
            best_error = param_history[-1]["error_threshold"]
            print(f"Smallest tested error threshold: {best_error}")
        else:
            best_error = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(is_converged), best_error, cost_history, param_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal parameters for heat_steady_2d simulation")

    # Search mode selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["dx", "relax", "T_init", "error_threshold"],
        required=True,
        help="Choose which parameter to search",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile")

    # Parameter values
    parser.add_argument("--dx", type=float, default=0.005, help="Grid spacing")
    parser.add_argument("--relax", type=float, default=1.0, help="Relaxation factor")
    parser.add_argument("--T_init", type=float, default=0.25, help="Initial temperature")
    parser.add_argument("--error_threshold", type=float, default=1e-7, help="Tolerance for inner convergence checking")

    # Fixed parameters
    parser.add_argument("--max_iter", type=int, default=20, help="Maximum number of iterations")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Tolerance for outter convergence check")

    args = parser.parse_args()

    if args.task == "dx":
        print("\n=== Starting grid resolution (dx) search ===")
        best_param, total_cost = find_optimal_dx(
            profile=args.profile,
            initial_dx=args.dx,
            relax=args.relax,
            error_threshold=args.error_threshold,
            T_init=args.T_init,
            tolerance=args.tolerance,
            max_iter=args.max_iter,
        )
        param_name = "dx"

    elif args.task == "relax":
        print("\n=== Starting relaxation factor search ===")
        # 0.1, 0.2, ..., 1.9 (defualt relax range)
        # NOTE apply a trick here (no need to try over-relaxation for Jacobi NOTE do not tell LLM!!!)
        relax_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        best_param, total_cost = grid_search_relax(
            profile=args.profile,
            dx=args.dx,
            relax_values=relax_values,
            error_threshold=args.error_threshold,
            T_init=args.T_init,
            max_iter=args.max_iter,
        )
        param_name = "relaxation factor"

    elif args.task == "T_init":
        print("\n=== Starting initial temperature search ===")
        # 0.05, 0.1, 0.15, ..., 0.9
        T_init_values = [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
        ]
        best_param, total_cost = grid_search_T_init(
            profile=args.profile,
            dx=args.dx,
            relax=args.relax,
            error_threshold=args.error_threshold,
            T_init_values=T_init_values,
            max_iter=args.max_iter,
        )
        param_name = "initial temperature"

    elif args.task == "error_threshold":
        print("\n=== Starting error threshold search ===")
        best_param, total_cost = find_optimal_error_threshold(
            profile=args.profile,
            dx=args.dx,
            relax=args.relax,
            T_init=args.T_init,
            initial_error=args.error_threshold,
            tolerance=args.tolerance,
            max_iter=args.max_iter,
        )
        param_name = "error threshold"

    if best_param is not None:
        print(f"\nRecommended {param_name}: {best_param}, total cost: {total_cost}")
    else:
        print(f"\nNo convergent {param_name} found within the given iterations, total cost: {total_cost}")
