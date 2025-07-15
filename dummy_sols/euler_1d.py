import argparse
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.euler_1d import run_sim_euler_1d, compare_res_euler_1d


def find_convergent_cfl(profile, cfl, beta, k, tolerance_linf, tolerance_rmse):
    """Iteratively reduce CFL number until convergence is achieved."""
    cfl_history = []
    cost_history = []
    param_history = []

    max_iter = 10  # Fixed maximum iterations

    current_cfl = cfl
    converged = False
    best_cfl = None

    for i in range(max_iter):
        print(f"\nRunning simulation with CFL = {current_cfl}, beta = {beta}, k = {k}")

        # Run simulation and load results
        cost_i = run_sim_euler_1d(profile, current_cfl, beta, k)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)
        param_history.append({"cfl": current_cfl, "beta": beta, "k": k})

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, linf_norm, rmse = compare_res_euler_1d(
                profile, prev_cfl, beta, k, profile, current_cfl, beta, k, tolerance_linf, tolerance_rmse
            )

            if is_converged:
                print(f"Convergence achieved between CFL {prev_cfl} and {current_cfl}")
                best_cfl = prev_cfl  # The larger of the two CFLs that converged
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

    return bool(converged), best_cfl, cost_history, param_history


def find_optimal_beta(profile, k, tolerance_linf, tolerance_rmse):
    """
    Grid search over beta ∈ [0.5, 2.0] (step 0.1) for optimal limiter parameter.

    Returns
    -------
    is_converged_optimal : bool
        Whether optimal_beta + optimal_cfl simulation converged.
    optimal_param : (float | None, float | None)
        (optimal_beta, optimal_cfl). None if no convergent solution found.
    optimal_cost_history : list[float] | None
        Cost history for optimal_beta corresponding to converged CFL sequence.
        None if no convergent solution found.
    param_history : list
        Full parameter exploration history.
    """
    beta_values = np.linspace(0.5, 2.0, 16)  # [0.5, 0.6, ..., 2.0]
    param_history = []  # Save (beta, cfl_history)
    beta_results = []  # Save key info for each beta (when converged)

    for beta in beta_values:
        beta = round(float(beta), 1)
        print(f"\n=== Testing beta = {beta} ===")

        is_converged, best_cfl, cost_history, one_param_history = find_convergent_cfl(
            profile, 1.0, beta, k, tolerance_linf, tolerance_rmse
        )

        # Record CFL exploration trajectory for each beta
        param_history.append(one_param_history)

        # If convergent CFL found, save to results pool
        if best_cfl is not None:
            total_cost = sum(cost_history[:-1])  # Calculate total cost
            beta_results.append(
                {
                    "beta": beta,
                    "best_cfl": best_cfl,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,  # Save complete cost_history
                }
            )
            print(f"beta = {beta}: Best CFL = {best_cfl}, Total Cost = {total_cost}")
        else:
            print(f"beta = {beta}: No convergent CFL found")

    # Select convergent solution with minimum total cost
    if beta_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in beta_results]))
        opt_rec = beta_results[min_cost_idx]

        optimal_beta = opt_rec["beta"]
        optimal_cfl = opt_rec["best_cfl"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal beta found: {optimal_beta} with CFL = {optimal_cfl}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_beta = optimal_cfl = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal beta found")

    optimal_param = (optimal_beta, optimal_cfl)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


def find_optimal_k(profile, beta, tolerance_linf, tolerance_rmse):
    """
    Grid search over k ∈ [-1, 1] (step 0.1) for optimal blending parameter.

    Returns
    -------
    is_converged_optimal : bool
        Whether optimal_k + optimal_cfl simulation converged.
    optimal_param : (float | None, float | None)
        (optimal_k, optimal_cfl). None if no convergent solution found.
    optimal_cost_history : list[float] | None
        Cost history for optimal_k corresponding to converged CFL sequence.
        None if no convergent solution found.
    param_history : list
        Full parameter exploration history.
    """
    k_values = np.linspace(-1.0, 1.0, 21)  # [-1.0, -0.9, ..., 1.0]
    param_history = []
    k_results = []  # Save key info for each k (when converged)

    for k in k_values:
        k = round(float(k), 1)
        print(f"\n=== Testing k = {k} ===")

        is_converged, best_cfl, cost_history, one_param_history = find_convergent_cfl(
            profile, 1.0, beta, k, tolerance_linf, tolerance_rmse
        )

        # Record CFL exploration trajectory for each k
        param_history.append(one_param_history)

        # If convergent CFL found, save to results pool
        if best_cfl is not None:
            total_cost = sum(cost_history[:-1])  # Calculate total cost
            k_results.append(
                {
                    "k": k,
                    "best_cfl": best_cfl,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal parameters for Euler 1D simulation")

    # Search mode selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["cfl", "beta", "k"],
        required=True,
        help="Choose which parameter to search: 'cfl', 'beta', or 'k'",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile configuration")

    # Controllable parameters
    parser.add_argument("--cfl", type=float, default=1.0, help="Initial CFL number to start testing")
    parser.add_argument("--beta", type=float, default=1.0, help="Limiter parameter for generalized superbee limiter")
    parser.add_argument(
        "--k", type=float, default=1.0, help="Blending parameter between central (1) and upwind (-1) scheme"
    )

    # Tolerance parameters
    parser.add_argument("--tolerance_linf", type=float, default=0.2, help="Linf tolerance for convergence checking")
    parser.add_argument("--tolerance_rmse", type=float, default=0.02, help="RMSE tolerance for convergence checking")

    args = parser.parse_args()

    if args.task == "cfl":
        print("\n=== Starting CFL convergence search ===")
        is_converged, best_cfl, cost_history, param_history = find_convergent_cfl(
            profile=args.profile,
            cfl=args.cfl,
            beta=args.beta,
            k=args.k,
            tolerance_linf=args.tolerance_linf,
            tolerance_rmse=args.tolerance_rmse,
        )

        if best_cfl is not None:
            print(f"\nRecommended CFL: {best_cfl}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent CFL found, total cost: {sum(cost_history)}")

    elif args.task == "beta":
        print("\n=== Starting beta parameter search ===")
        is_converged, optimal_param, optimal_cost_history, param_history = find_optimal_beta(
            profile=args.profile,
            k=args.k,
            tolerance_linf=args.tolerance_linf,
            tolerance_rmse=args.tolerance_rmse,
        )

        optimal_beta, optimal_cfl = optimal_param
        if optimal_beta is not None:
            print(f"\nRecommended beta: {optimal_beta} with CFL: {optimal_cfl}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal beta found")

    elif args.task == "k":
        print("\n=== Starting k parameter search ===")
        is_converged, optimal_param, optimal_cost_history, param_history = find_optimal_k(
            profile=args.profile,
            beta=args.beta,
            tolerance_linf=args.tolerance_linf,
            tolerance_rmse=args.tolerance_rmse,
        )

        optimal_k, optimal_cfl = optimal_param
        if optimal_k is not None:
            print(f"\nRecommended k: {optimal_k} with CFL: {optimal_cfl}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal k found")
