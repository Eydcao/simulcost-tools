import argparse
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.burgers_1d import (
    run_sim_burgers_1d,
    compare_res_burgers_1d,
)


def find_convergent_cfl(profile, cfl, k, beta, tolerance_rmse, multiplication_factor, max_iter):
    """Iteratively reduce CFL number until convergence is achieved."""
    cfl_history = []
    cost_history = []
    param_history = []

    current_cfl = cfl
    converged = False
    best_cfl = None

    for i in range(max_iter):
        print(f"\nRunning simulation with CFL = {current_cfl}, k = {k}, beta = {beta}")

        # Run simulation and load results (use default n_space from config)
        cost_i = run_sim_burgers_1d(profile, current_cfl, k, beta, 2048)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)
        param_history.append({"cfl": current_cfl, "k": k, "beta": beta})

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, rmse = compare_res_burgers_1d(
                profile, prev_cfl, k, beta, profile, current_cfl, k, beta, tolerance_rmse, 2048, 2048
            )

            if is_converged:
                print(f"Convergence achieved between CFL {prev_cfl} and {current_cfl}")
                best_cfl = cfl_history[-1]  # The larger of the two CFLs that converged
                converged = True
                break
            else:
                print(f"No convergence between CFL {prev_cfl} and {current_cfl}")

        # Prepare next CFL (half of current)
        next_cfl = current_cfl * multiplication_factor
        current_cfl = next_cfl

    if converged:
        print(f"\nConvergent CFL found: {best_cfl}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(cfl_history) > 1:
            best_cfl = cfl_history[-1]
            print(f"Smallest tested CFL: {best_cfl}")
        else:
            best_cfl = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(is_converged), best_cfl, cost_history, param_history


def find_optimal_k(
    profile,
    cfl,
    beta,
    n_space,
    tolerance_rmse,
    search_range_min,
    search_range_max,
    search_range_slice_num,
    multiplication_factor,
    max_iter,
):
    """
    Grid search on k ∈ [-1, 1] (step size 0.1) and record all n_space exploration sequences for each k.

    Returns
    -------
    is_converged_optimal : bool
        Whether the simulation with optimal_k + optimal_n_space finally converged.
    optimal_param : (float | None, float | None)
        (optimal_k, optimal_n_space). Both are None if no convergent solution exists.
    optimal_cost_history : list[float] | None
        The n_space → cost evolution sequence (cost_history) corresponding to optimal_k and finally converged.
        None if no convergent solution exists.
    param_history : list
    """
    k_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
    param_history = []  # Store (k, n_space_history)
    k_results = []  # Store key information for each k (when converged)

    for k in k_values:
        k = round(float(k), 1)
        print(f"\n=== Testing k = {k} ===")

        is_converged, best_n_space, cost_history, one_param_history = find_convergent_n_space(
            profile, n_space, cfl, k, beta, tolerance_rmse, multiplication_factor, max_iter
        )

        # Record n_space exploration trajectory for each k
        param_history.append(one_param_history)

        # If convergent n_space is found, store in result pool
        if best_n_space is not None:
            total_cost = sum(cost_history)  # Calculate total cost
            k_results.append(
                {
                    "k": k,
                    "best_n_space": best_n_space,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,  # ★ Key: save complete cost_history
                }
            )
            print(f"k = {k}: Best n_space = {best_n_space}, Total Cost = {total_cost}")
        else:
            print(f"k = {k}: No convergent n_space found")

    # Select convergent solution with minimum total cost
    if k_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in k_results]))
        opt_rec = k_results[min_cost_idx]

        optimal_k = opt_rec["k"]
        optimal_n_space = opt_rec["best_n_space"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal k found: {optimal_k} with n_space = {optimal_n_space}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_k = optimal_n_space = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal k found")

    optimal_param = (optimal_k, optimal_n_space)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


def find_optimal_beta(
    profile,
    cfl,
    k,
    n_space,
    tolerance_rmse,
    search_range_min,
    search_range_max,
    search_range_slice_num,
    multiplication_factor,
    max_iter,
):
    """
    Grid search on beta ∈ [1.0, 2.0] and record all n_space exploration sequences for each beta.

    Returns
    -------
    is_converged_optimal : bool
        Whether the simulation with optimal_beta + optimal_n_space finally converged.
    optimal_param : (float | None, float | None)
        (optimal_beta, optimal_n_space). Both are None if no convergent solution exists.
    optimal_cost_history : list[float] | None
        The n_space → cost evolution sequence (cost_history) corresponding to optimal_beta and finally converged.
        None if no convergent solution exists.
    param_history : list
    """
    beta_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
    param_history = []
    beta_results = []  # Store key information for each beta when converged

    for beta in beta_values:
        beta = round(float(beta), 1)
        print(f"\n=== Testing beta = {beta} ===")

        is_converged, best_n_space, cost_history, one_param_history = find_convergent_n_space(
            profile, n_space, cfl, k, beta, tolerance_rmse, multiplication_factor, max_iter
        )

        # Record n_space exploration trajectory for each beta
        param_history.append(one_param_history)

        # If convergent n_space is found, collect results
        if best_n_space is not None:
            total_cost = sum(cost_history)  # Calculate total cost
            beta_results.append(
                {
                    "beta": beta,
                    "best_n_space": best_n_space,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
                }
            )
            print(f"beta = {beta}: Best n_space = {best_n_space}, Total Cost = {total_cost}")
        else:
            print(f"beta = {beta}: No convergent n_space found")

    # Select convergent solution with minimum total cost
    if beta_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in beta_results]))
        opt_rec = beta_results[min_cost_idx]

        optimal_beta = opt_rec["beta"]
        optimal_n_space = opt_rec["best_n_space"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal beta found: {optimal_beta} with n_space = {optimal_n_space}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_beta = optimal_n_space = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal beta found")

    optimal_param = (optimal_beta, optimal_n_space)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


def find_convergent_n_space(profile, n_space, cfl, k, beta, tolerance_rmse, multiplication_factor, max_iter):
    """Iteratively increase n_space (spatial resolution) until convergence is achieved."""
    n_space_history = []
    cost_history = []
    param_history = []

    current_n_space = n_space
    converged = False
    best_n_space = None

    for i in range(max_iter):
        print(f"\nRunning simulation with n_space = {current_n_space}, cfl = {cfl}, k = {k}, beta = {beta}")

        # Run simulation with n_space parameter
        cost_i = run_sim_burgers_1d(profile, cfl, k, beta, current_n_space)
        cost_history.append(cost_i)
        n_space_history.append(current_n_space)
        param_history.append({"n_space": current_n_space, "cfl": cfl, "k": k, "beta": beta})

        # If we have previous results to compare with
        if len(n_space_history) > 1:
            prev_n_space = n_space_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, rmse = compare_res_burgers_1d(
                profile, cfl, k, beta, profile, cfl, k, beta, tolerance_rmse, prev_n_space, current_n_space
            )

            if is_converged:
                print(f"Convergence achieved between n_space {prev_n_space} and {current_n_space}")
                best_n_space = n_space_history[-1]  # The finer resolution that converged
                converged = True
                break
            else:
                print(f"No convergence between n_space {prev_n_space} and {current_n_space}")

        # Prepare next n_space (double current)
        next_n_space = current_n_space * multiplication_factor
        current_n_space = next_n_space

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

    return bool(converged), best_n_space, cost_history, param_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal parameters for Burgers 1D simulation")

    # Search mode selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["cfl", "k", "beta", "n_space"],
        required=True,
        help="Choose which parameter to search: 'cfl', 'k', 'beta', or 'n_space'",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile configuration")

    # Controllable parameter
    parser.add_argument("--cfl", type=float, default=1.0, help="Initial CFL number to start testing")
    parser.add_argument(
        "--k", type=float, default=0, help="The blending parameters between central (1) and upwind (-1) scheme"
    )
    parser.add_argument("--beta", type=float, default=1.0, help="beta parameter for generalized superbee limiter")
    parser.add_argument(
        "--n_space", type=int, default=256, help="Initial n_space (spatial resolution) to start testing"
    )

    # Tolerance parameter
    parser.add_argument("--tolerance_rmse", type=float, default=5e-3, help="RMSE tolerance for convergence checking")

    args = parser.parse_args()

    if args.task == "cfl":
        print("\n=== Starting CFL convergence search ===")
        is_converged, best_cfl, cost_history, param_history = find_convergent_cfl(
            profile=args.profile,
            cfl=args.cfl,
            k=args.k,
            beta=args.beta,
            tolerance_rmse=args.tolerance_rmse,
            multiplication_factor=0.5,
            max_iter=7,
        )

        if best_cfl is not None:
            print(f"\nRecommended CFL: {best_cfl}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent CFL found, total cost: {sum(cost_history)}")

    elif args.task == "k":
        print("\n=== Starting k parameter search ===")
        is_converged, optimal_param, optimal_cost_history, param_history = find_optimal_k(
            profile=args.profile,
            cfl=args.cfl,
            beta=args.beta,
            n_space=args.n_space,
            tolerance_rmse=args.tolerance_rmse,
            search_range_min=-1.0,
            search_range_max=1.0,
            search_range_slice_num=11,
            multiplication_factor=2,
            max_iter=7,
        )

        optimal_k, optimal_n_space = optimal_param
        if optimal_k is not None:
            print(f"\nRecommended k: {optimal_k} with n_space: {optimal_n_space}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal k found")

    elif args.task == "beta":
        print("\n=== Starting beta parameter search ===")
        is_converged, optimal_param, optimal_cost_history, param_history = find_optimal_beta(
            profile=args.profile,
            cfl=args.cfl,
            k=args.k,
            n_space=args.n_space,
            tolerance_rmse=args.tolerance_rmse,
            search_range_min=1.0,
            search_range_max=2.0,
            search_range_slice_num=6,
            multiplication_factor=2,
            max_iter=7,
        )

        optimal_beta, optimal_n_space = optimal_param
        if optimal_beta is not None:
            print(f"\nRecommended beta: {optimal_beta} with n_space: {optimal_n_space}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal beta found")

    elif args.task == "n_space":
        print("\n=== Starting n_space convergence search ===")
        is_converged, best_n_space, cost_history, param_history = find_convergent_n_space(
            profile=args.profile,
            n_space=args.n_space,
            cfl=args.cfl,
            k=args.k,
            beta=args.beta,
            tolerance_rmse=args.tolerance_rmse,
            multiplication_factor=2,
            max_iter=6,
        )

        if best_n_space is not None:
            print(f"\nRecommended n_space: {best_n_space}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent n_space found, total cost: {sum(cost_history) if cost_history else 0}")
