import argparse
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.burgers_1d import run_sim_burgers_1d, compare_res_burgers_1d


def find_convergent_cfl(profile, cfl, k, w, tolerance_infity, tolerance_2):
    """Iteratively reduce CFL number until convergence is achieved."""
    cfl_history = []
    cost_history = []
    param_history = []

    max_iter = 10  # Fixed maximum iterations

    current_cfl = cfl
    converged = False
    best_cfl = None

    for i in range(max_iter):
        print(f"\nRunning simulation with CFL = {current_cfl}, k = {k}, w = {w}")

        # Run simulation and load results
        cost_i = run_sim_burgers_1d(profile, current_cfl, k, w)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)
        param_history.append({"cfl": current_cfl, "k": k, "w": w})

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, linf_norm, rmse = compare_res_burgers_1d(
                profile, prev_cfl, k, w, profile, current_cfl, k, w, tolerance_infity, tolerance_2
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


def find_optimal_k(profile, w, tolerance_infity, tolerance_2):
    """
    Grid search on k ∈ [-1, 1] (step size 0.1) and record all CFL exploration sequences for each k.

    Returns
    -------
    is_converged_optimal : bool
        Whether the simulation with optimal_k + optimal_cfl finally converged.
    optimal_param : (float | None, float | None)
        (optimal_k, optimal_cfl). Both are None if no convergent solution exists.
    optimal_cost_history : list[float] | None
        The CFL → cost evolution sequence (cost_history) corresponding to optimal_k and finally converged.
        None if no convergent solution exists.
    param_history : list
    """
    k_values = np.linspace(-1.0, 1.0, 21)
    param_history = []  # Store (k, cfl_history)
    k_results = []  # Store key information for each k (when converged)

    for k in k_values:
        k = round(float(k), 1)
        print(f"\n=== Testing k = {k} ===")

        is_converged, best_cfl, cost_history, one_param_history = find_convergent_cfl(
            profile, 1.0, k, w, tolerance_infity, tolerance_2
        )

        # Record CFL exploration trajectory for each k
        param_history.append(one_param_history)

        # If convergent CFL is found, store in result pool
        if best_cfl is not None:
            total_cost = sum(cost_history[:-1])  # Calculate total cost
            k_results.append(
                {
                    "k": k,
                    "best_cfl": best_cfl,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,  # ★ Key: save complete cost_history
                }
            )
            print(f"k = {k}: Best CFL = {best_cfl}, Total Cost = {total_cost}")
        else:
            print(f"k = {k}: No convergent CFL found")

    # Select convergent solution with minimum total cost
    if k_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in k_results]))
        opt_rec = k_results[min_cost_idx]

        optimal_k = opt_rec["k"]
        optimal_cfl = opt_rec["best_cfl"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal k found: {optimal_k} with CFL = {optimal_cfl}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_k = optimal_cfl = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal k found")

    optimal_param = (optimal_k, optimal_cfl)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


def find_optimal_w(profile, k, tolerance_infity, tolerance_2):
    """
    Grid search on w ∈ [0, 2] (step size 0.1) and record all CFL exploration sequences for each w.

    Returns
    -------
    is_converged_optimal : bool
        Whether the simulation with optimal_w + optimal_cfl finally converged.
    optimal_param : (float | None, float | None)
        (optimal_w, optimal_cfl). Both are None if no convergent solution exists.
    optimal_cost_history : list[float] | None
        The CFL → cost evolution sequence (cost_history) corresponding to optimal_w and finally converged.
        None if no convergent solution exists.
    param_history : list
    """
    w_values = np.linspace(0.0, 2.0, 21)
    param_history = []
    w_results = []  # Store key information for each w when converged

    for w in w_values:
        w = round(float(w), 1)
        print(f"\n=== Testing w = {w} ===")

        is_converged, best_cfl, cost_history, one_param_history = find_convergent_cfl(
            profile, 1.0, k, w, tolerance_infity, tolerance_2
        )

        # Record CFL exploration trajectory for each w
        param_history.append(one_param_history)

        # If convergent CFL is found, collect results
        if best_cfl is not None:
            total_cost = sum(cost_history[:-1])  # Calculate total cost
            w_results.append(
                {
                    "w": w,
                    "best_cfl": best_cfl,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
                }
            )
            print(f"w = {w}: Best CFL = {best_cfl}, Total Cost = {total_cost}")
        else:
            print(f"w = {w}: No convergent CFL found")

    # Select convergent solution with minimum total cost
    if w_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in w_results]))
        opt_rec = w_results[min_cost_idx]

        optimal_w = opt_rec["w"]
        optimal_cfl = opt_rec["best_cfl"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal w found: {optimal_w} with CFL = {optimal_cfl}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_w = optimal_cfl = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal w found")

    optimal_param = (optimal_w, optimal_cfl)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal parameters for Burgers 1D simulation")

    # Search mode selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["cfl", "k", "w"],
        required=True,
        help="Choose which parameter to search: 'cfl', 'k', or 'w'",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile configuration")

    # Controllable parameter
    parser.add_argument("--cfl", type=float, default=1.0, help="Initial CFL number to start testing")
    parser.add_argument(
        "--k", type=float, default=0, help="The blending parameters between central (1) and upwind (-1) scheme"
    )
    parser.add_argument("--w", type=float, default=1.0, help="w parameter for minmod limiter")

    # Tolerance parameter
    parser.add_argument("--tolerance_infity", type=float, default=5e-2, help="Tolerance for convergence checking")
    parser.add_argument("--tolerance_2", type=float, default=5e-3, help="Tolerance for convergence checking")

    args = parser.parse_args()

    if args.task == "cfl":
        print("\n=== Starting CFL convergence search ===")
        best_cfl, total_cost, cost_history, param_history = find_convergent_cfl(
            profile=args.profile,
            cfl=args.cfl,
            k=args.k,
            w=args.w,
            tolerance_infity=args.tolerance_infity,
            tolerance_2=args.tolerance_2,
        )

        if best_cfl is not None:
            print(f"\nRecommended CFL: {best_cfl}, total cost: {total_cost}")
        else:
            print(f"\nNo convergent CFL found, total cost: {total_cost}")

    elif args.task == "k":
        print("\n=== Starting k parameter search ===")
        optimal_k, optimal_cfl, optimal_cost, k_results = find_optimal_k(
            profile=args.profile,
            w=args.w,
            tolerance_infity=args.tolerance_infity,
            tolerance_2=args.tolerance_2,
        )

        if optimal_k is not None:
            print(f"\nRecommended k: {optimal_k} with CFL: {optimal_cfl}")
            print(f"Total cost across all k tests: {optimal_cost}")
            print(f"Print log for each k")
            for k_log in k_results:
                print(k_log)
        else:
            print("\nNo optimal k found")

    elif args.task == "w":
        print("\n=== Starting w parameter search ===")
        optimal_w, optimal_cfl, optimal_cost, w_results = find_optimal_w(
            profile=args.profile,
            k=args.k,
            tolerance_infity=args.tolerance_infity,
            tolerance_2=args.tolerance_2,
        )

        if optimal_w is not None:
            print(f"\nRecommended w: {optimal_w} with CFL: {optimal_cfl}")
            print(f"Total cost across all w tests: {optimal_cost}")
            print(f"Print log for each w")
            for w_log in w_results:
                print(w_log)
        else:
            print("\nNo optimal w found")
