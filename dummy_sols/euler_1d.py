import argparse
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.euler_1d import run_sim_euler_1d, compare_res_euler_1d


def find_convergent_cfl(profile, cfl, beta, k, n_space, tolerance_linf, tolerance_rmse):
    """Iteratively reduce CFL number until convergence is achieved."""
    cfl_history = []
    cost_history = []
    param_history = []

    max_iter = 7  # Fixed maximum iterations

    current_cfl = cfl
    converged = False
    best_cfl = None

    for i in range(max_iter):
        print(f"\nRunning simulation with CFL = {current_cfl}, beta = {beta}, k = {k}, n_space = {n_space}")

        # Run simulation and load results
        cost_i = run_sim_euler_1d(profile, current_cfl, beta, k, n_space)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)
        param_history.append({"cfl": current_cfl, "beta": beta, "k": k, "n_space": n_space})

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, linf_norm, rmse = compare_res_euler_1d(
                profile,
                prev_cfl,
                beta,
                k,
                profile,
                current_cfl,
                beta,
                k,
                tolerance_linf,
                tolerance_rmse,
                n_space,
                n_space,
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

    return bool(converged), best_cfl, cost_history, param_history


def find_optimal_beta(profile, cfl, k, n_space, tolerance_linf, tolerance_rmse):
    """
    Grid search over beta ∈ [1.0, 2.0] (step 0.1) for optimal limiter parameter.
    For each beta, iterate n_space until spatial convergence is achieved.

    Returns
    -------
    is_converged_optimal : bool
        Whether optimal_beta achieved spatial convergence.
    optimal_param : (float | None, int | None)
        (optimal_beta, optimal_n_space). None if no convergent solution found.
    optimal_cost_history : list[float] | None
        Cost history for optimal_beta corresponding to converged n_space sequence.
        None if no convergent solution found.
    param_history : list
        Full parameter exploration history.
    """
    beta_values = np.linspace(1.0, 2.0, 6)  # [1.0, 1.2, ..., 2.0]
    param_history = []
    beta_results = []  # Save key info for each beta (when converged)

    for beta in beta_values:
        beta = round(float(beta), 1)
        print(f"\n=== Testing beta = {beta} ===")

        is_converged, best_n_space, cost_history, one_param_history = find_convergent_n_space(
            profile, cfl, n_space, beta, k, tolerance_linf, tolerance_rmse
        )

        # Record n_space exploration trajectory for each beta
        param_history.append(one_param_history)

        # If convergent n_space found, save to results pool
        if best_n_space is not None:
            total_cost = sum(cost_history)
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


def find_optimal_k(profile, cfl, beta, n_space, tolerance_linf, tolerance_rmse):
    """
    Grid search over k ∈ [-1, 1] (step 0.1) for optimal blending parameter.
    For each k, iterate n_space until spatial convergence is achieved.

    Returns
    -------
    is_converged_optimal : bool
        Whether optimal_k achieved spatial convergence.
    optimal_param : (float | None, int | None)
        (optimal_k, optimal_n_space). None if no convergent solution found.
    optimal_cost_history : list[float] | None
        Cost history for optimal_k corresponding to converged n_space sequence.
        None if no convergent solution found.
    param_history : list
        Full parameter exploration history.
    """
    k_values = np.linspace(-1.0, 1.0, 11)  # [-1.0, -0.8, ..., 1.0]
    param_history = []
    k_results = []  # Save key info for each k (when converged)

    for k in k_values:
        k = round(float(k), 1)
        print(f"\n=== Testing k = {k} ===")

        is_converged, best_n_space, cost_history, one_param_history = find_convergent_n_space(
            profile, cfl, n_space, beta, k, tolerance_linf, tolerance_rmse
        )

        # Record n_space exploration trajectory for each k
        param_history.append(one_param_history)

        # If convergent n_space found, save to results pool
        if best_n_space is not None:
            total_cost = sum(cost_history)
            k_results.append(
                {
                    "k": k,
                    "best_n_space": best_n_space,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
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


def find_convergent_n_space(profile, cfl, n_space, beta, k, tolerance_linf, tolerance_rmse):
    """Iteratively increase n_space (decrease dx) until convergence is achieved with fixed CFL."""
    n_space_history = []
    cost_history = []
    param_history = []

    max_iter = 6  # Fixed maximum iterations (n_space can grow quickly)

    current_n_space = n_space
    converged = False
    best_n_space = None

    for i in range(max_iter):
        print(f"\nRunning simulation with n_space = {current_n_space}, CFL = {cfl}, beta = {beta}, k = {k}")

        # Run simulation with fixed CFL
        cost_i = run_sim_euler_1d(profile, cfl, beta, k, current_n_space)
        cost_history.append(cost_i)
        n_space_history.append(current_n_space)
        param_history.append({"n_space": current_n_space, "cfl": cfl, "beta": beta, "k": k})

        # If we have previous results to compare with
        if len(n_space_history) > 1:
            prev_n_space = n_space_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, linf_norm, rmse = compare_res_euler_1d(
                profile,
                cfl,
                beta,
                k,
                profile,
                cfl,
                beta,
                k,
                tolerance_linf,
                tolerance_rmse,
                prev_n_space,
                current_n_space,
            )

            if is_converged:
                print(f"Convergence achieved between n_space {prev_n_space} and {current_n_space}")
                best_n_space = n_space_history[-1]  # The finer grid that converged
                converged = True
                break
            else:
                print(f"No convergence between n_space {prev_n_space} and {current_n_space}")

        # Prepare next n_space (double current for geometric progression)
        next_n_space = current_n_space * 2
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
    parser = argparse.ArgumentParser(description="Find optimal parameters for Euler 1D simulation")

    # Search mode selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["cfl", "beta", "k", "n_space"],
        required=True,
        help="Choose which parameter to search: 'cfl', 'beta', 'k', or 'n_space'",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile configuration")

    # Controllable parameters
    parser.add_argument("--cfl", type=float, default=0.25, help="Initial CFL number to start testing")
    parser.add_argument("--beta", type=float, default=1.0, help="Limiter parameter for generalized superbee limiter")
    parser.add_argument(
        "--k", type=float, default=-1.0, help="Blending parameter between central (1) and upwind (-1) scheme"
    )
    parser.add_argument(
        "--n_space", type=int, default=256, help="Initial number of grid cells for spatial discretization"
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
            n_space=args.n_space,
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
            cfl=args.cfl,
            k=args.k,
            n_space=args.n_space,
            tolerance_linf=args.tolerance_linf,
            tolerance_rmse=args.tolerance_rmse,
        )

        optimal_beta, optimal_n_space = optimal_param
        if optimal_beta is not None:
            print(f"\nRecommended beta: {optimal_beta} with n_space: {optimal_n_space}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal beta found")

    elif args.task == "k":
        print("\n=== Starting k parameter search ===")
        is_converged, optimal_param, optimal_cost_history, param_history = find_optimal_k(
            profile=args.profile,
            cfl=args.cfl,
            beta=args.beta,
            n_space=args.n_space,
            tolerance_linf=args.tolerance_linf,
            tolerance_rmse=args.tolerance_rmse,
        )

        optimal_k, optimal_n_space = optimal_param
        if optimal_k is not None:
            print(f"\nRecommended k: {optimal_k} with n_space: {optimal_n_space}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal k found")

    elif args.task == "n_space":
        print("\n=== Starting n_space convergence search ===")
        is_converged, best_n_space, cost_history, param_history = find_convergent_n_space(
            profile=args.profile,
            cfl=args.cfl,
            n_space=args.n_space,
            beta=args.beta,
            k=args.k,
            tolerance_linf=args.tolerance_linf,
            tolerance_rmse=args.tolerance_rmse,
        )

        if best_n_space is not None:
            print(f"\nRecommended n_space: {best_n_space}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent n_space found, total cost: {sum(cost_history)}")
