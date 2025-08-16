import argparse
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.burgers_1d import (
    run_sim_burgers_1d,
    compare_res_burgers_1d,
)


def find_convergent_cfl(profile, cfl, k, w, tolerance_infity, tolerance_2, multiplication_factor, max_iter):
    """Iteratively reduce CFL number until convergence is achieved."""
    cfl_history = []
    cost_history = []
    param_history = []

    current_cfl = cfl
    converged = False
    best_cfl = None

    for i in range(max_iter):
        print(f"\nRunning simulation with CFL = {current_cfl}, k = {k}, w = {w}")

        # Run simulation and load results (use default n_space from config)
        cost_i = run_sim_burgers_1d(profile, current_cfl, k, w, 2048)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)
        param_history.append({"cfl": current_cfl, "k": k, "w": w})

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, linf_norm, rmse = compare_res_burgers_1d(
                profile, prev_cfl, k, w, profile, current_cfl, k, w, tolerance_infity, tolerance_2, 2048, 2048
            )

            if is_converged:
                print(f"Convergence achieved between CFL {prev_cfl} and {current_cfl}")
                best_cfl = cfl_history[-1]  # The larger of the two CFLs that converged
                converged = True
                break
            else:
                print(f"No convergence between CFL {prev_cfl} and {current_cfl}")

        # Prepare next CFL (half of current)
        next_cfl = current_cfl / 2
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


def find_optimal_k(profile, cfl, w, n_space, tolerance_infity, tolerance_2, search_range_min, search_range_max, search_range_slice_num, multiplication_factor, max_iter):
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
            profile, n_space, cfl, k, w, tolerance_infity, tolerance_2, multiplication_factor, max_iter
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


def find_optimal_w(profile, cfl, k, n_space, tolerance_infity, tolerance_2, search_range_min, search_range_max, search_range_slice_num, multiplication_factor, max_iter):
    """
    Grid search on w ∈ [0, 2] (step size 0.1) and record all n_space exploration sequences for each w.

    Returns
    -------
    is_converged_optimal : bool
        Whether the simulation with optimal_w + optimal_n_space finally converged.
    optimal_param : (float | None, float | None)
        (optimal_w, optimal_n_space). Both are None if no convergent solution exists.
    optimal_cost_history : list[float] | None
        The n_space → cost evolution sequence (cost_history) corresponding to optimal_w and finally converged.
        None if no convergent solution exists.
    param_history : list
    """
    w_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
    param_history = []
    w_results = []  # Store key information for each w when converged

    for w in w_values:
        w = round(float(w), 1)
        print(f"\n=== Testing w = {w} ===")

        is_converged, best_n_space, cost_history, one_param_history = find_convergent_n_space(
            profile, n_space, cfl, k, w, tolerance_infity, tolerance_2, multiplication_factor, max_iter
        )

        # Record n_space exploration trajectory for each w
        param_history.append(one_param_history)

        # If convergent n_space is found, collect results
        if best_n_space is not None:
            total_cost = sum(cost_history)  # Calculate total cost
            w_results.append(
                {
                    "w": w,
                    "best_n_space": best_n_space,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
                }
            )
            print(f"w = {w}: Best n_space = {best_n_space}, Total Cost = {total_cost}")
        else:
            print(f"w = {w}: No convergent n_space found")

    # Select convergent solution with minimum total cost
    if w_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in w_results]))
        opt_rec = w_results[min_cost_idx]

        optimal_w = opt_rec["w"]
        optimal_n_space = opt_rec["best_n_space"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal w found: {optimal_w} with n_space = {optimal_n_space}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_w = optimal_n_space = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal w found")

    optimal_param = (optimal_w, optimal_n_space)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


def find_convergent_n_space(profile, n_space, cfl, k, w, tolerance_infity, tolerance_2, multiplication_factor, max_iter):
    """Iteratively increase n_space (spatial resolution) until convergence is achieved."""
    n_space_history = []
    cost_history = []
    param_history = []

    current_n_space = n_space
    converged = False
    best_n_space = None

    for i in range(max_iter):
        print(f"\nRunning simulation with n_space = {current_n_space}, cfl = {cfl}, k = {k}, w = {w}")

        # Run simulation with n_space parameter
        cost_i = run_sim_burgers_1d(profile, cfl, k, w, current_n_space)
        cost_history.append(cost_i)
        n_space_history.append(current_n_space)
        param_history.append({"n_space": current_n_space, "cfl": cfl, "k": k, "w": w})

        # If we have previous results to compare with
        if len(n_space_history) > 1:
            prev_n_space = n_space_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, linf_norm, rmse = compare_res_burgers_1d(
                profile, cfl, k, w, profile, cfl, k, w, tolerance_infity, tolerance_2, prev_n_space, current_n_space
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
        choices=["cfl", "k", "w", "n_space"],
        required=True,
        help="Choose which parameter to search: 'cfl', 'k', 'w', or 'n_space'",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile configuration")

    # Controllable parameter
    parser.add_argument("--cfl", type=float, default=1.0, help="Initial CFL number to start testing")
    parser.add_argument(
        "--k", type=float, default=0, help="The blending parameters between central (1) and upwind (-1) scheme"
    )
    parser.add_argument("--w", type=float, default=1.0, help="w parameter for minmod limiter")
    parser.add_argument(
        "--n_space", type=int, default=256, help="Initial n_space (spatial resolution) to start testing"
    )

    # Tolerance parameter
    parser.add_argument("--tolerance_infity", type=float, default=5e-2, help="Tolerance for convergence checking")
    parser.add_argument("--tolerance_2", type=float, default=5e-3, help="Tolerance for convergence checking")

    args = parser.parse_args()

    if args.task == "cfl":
        print("\n=== Starting CFL convergence search ===")
        is_converged, best_cfl, cost_history, param_history = find_convergent_cfl(
            profile=args.profile,
            cfl=args.cfl,
            k=args.k,
            w=args.w,
            tolerance_infity=args.tolerance_infity,
            tolerance_2=args.tolerance_2,
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
            w=args.w,
            n_space=args.n_space,
            tolerance_infity=args.tolerance_infity,
            tolerance_2=args.tolerance_2,
        )

        optimal_k, optimal_n_space = optimal_param
        if optimal_k is not None:
            print(f"\nRecommended k: {optimal_k} with n_space: {optimal_n_space}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal k found")

    elif args.task == "w":
        print("\n=== Starting w parameter search ===")
        is_converged, optimal_param, optimal_cost_history, param_history = find_optimal_w(
            profile=args.profile,
            cfl=args.cfl,
            k=args.k,
            n_space=args.n_space,
            tolerance_infity=args.tolerance_infity,
            tolerance_2=args.tolerance_2,
        )

        optimal_w, optimal_n_space = optimal_param
        if optimal_w is not None:
            print(f"\nRecommended w: {optimal_w} with n_space: {optimal_n_space}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal w found")

    elif args.task == "n_space":
        print("\n=== Starting n_space convergence search ===")
        is_converged, best_n_space, cost_history, param_history = find_convergent_n_space(
            profile=args.profile,
            n_space=args.n_space,
            cfl=args.cfl,
            k=args.k,
            w=args.w,
            tolerance_infity=args.tolerance_infity,
            tolerance_2=args.tolerance_2,
        )

        if best_n_space is not None:
            print(f"\nRecommended n_space: {best_n_space}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent n_space found, total cost: {sum(cost_history) if cost_history else 0}")
