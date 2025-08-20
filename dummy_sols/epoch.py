import argparse
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.epoch import runEpoch, compare_res_epoch


def find_convergent_npart(profile, dt_multipler, nx, npart, field_order,particle_order, tolerance_rmse, multiplication_factor, max_iteration_num):
    """Iteratively increase npart (number pseudoparticles) until convergence is achieved with fixed parameters."""
    npart_history = []
    cost_history = []
    param_history = []

    current_npart = int(npart)
    converged = False
    best_npart = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with npart = {current_npart}, nx = {nx}, dt_mult = {dt_multipler}, field_Order = {field_order}, particle_order = {particle_order} ")

        # Run simulation with fixed CFL
        cost_i = runEpoch(profile, nx, dt_multipler, current_npart,field_order,particle_order)
        cost_history.append(cost_i)
        npart_history.append(current_npart)
        param_history.append({"nx": nx, "dt_mult":  dt_multipler, "npart": current_npart, "field_Order" : field_order, "particle_order" : particle_order})

        # If we have previous results to compare with
        if len(npart_history) > 1:
            prev_npart = npart_history[-2]

            # Compare with previous results
            is_converged = compare_res_epoch(
                profile,
                nx,
                dt_multipler,
                prev_npart,
                field_order,
                particle_order,
                profile,
                nx,
                dt_multipler,
                current_npart,
                field_order,
                particle_order,
                tolerance_rmse
            )

            if is_converged:
                print(f"Convergence achieved between npart {prev_npart} and {current_npart}")
                best_npart = npart_history[-1]  # The finer grid that converged
                converged = True
                break
            else:
                print(f"No convergence between npart {prev_npart} and {current_npart}")

        # Prepare next n_space using multiplication factor
        next_npart = int(current_npart * multiplication_factor)
        current_npart = next_npart

    if converged:
        print(f"\nConvergent npart found: {best_npart}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(npart_history) > 1:
            best_npart = npart_history[-1]
            print(f"Finest tested npart: {best_npart}")
        else:
            best_npart = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_npart, cost_history, param_history


def find_optimal_dt_multipler(
    profile, 
    nx, 
    npart, 
    field_order,
    particle_order,
    tolerance_rmse,
    search_range_min,
    search_range_max,
    search_range_slice_num,
    multiplication_factor,
    max_iteration_num,
):
    """
    Grid search over dt_multipler ∈ [search_range_min, search_range_max] for optimal limiter parameter.
    For each dt_multipler, iterate nx until spatial convergence is achieved.

    Returns
    -------
    is_converged_optimal : bool
        Whether optimal_dt_multipler achieved spatial convergence.
    optimal_param : (float | None, int | None)
        (optimal_dt_multipler, optimal_nx). None if no convergent solution found.
    optimal_cost_history : list[float] | None
        Cost history for optimal_dt_multipler corresponding to converged nx sequence.
        None if no convergent solution found.
    param_history : list
        Full parameter exploration history.
    """
    dt_multipler_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
    param_history = []
    dt_multipler_results = []  # Save key info for each dt_multipler (when converged)

    for dt_multipler in dt_multipler_values:
        dt_multipler = round(float(dt_multipler), 2)
        print(f"\n=== Testing dt_multipler = {dt_multipler} ===")

        is_converged, best_nx, cost_history, one_param_history = find_convergent_nx(
            profile, dt_multipler, nx,npart, field_order, particle_order, tolerance_rmse, multiplication_factor, max_iteration_num
        )

        # Record n_space exploration trajectory for each dt_multipler
        param_history.append(one_param_history)

        # If convergent n_space found, save to results pool
        if best_nx is not None:
            total_cost = sum(cost_history)
            dt_multipler_results.append(
                {
                    "dt_multipler": dt_multipler,
                    "best_nx": best_nx,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
                }
            )
            print(f"dt_multipler = {dt_multipler}: Best nx = {best_nx}, Total Cost = {total_cost}")
        else:
            print(f"dt_multipler = {dt_multipler}: No convergent nx found")

    # Select convergent solution with minimum total cost
    if dt_multipler_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in dt_multipler_results]))
        opt_rec = dt_multipler_results[min_cost_idx]

        optimal_dt_multipler = opt_rec["dt_multipler"]
        optimal_nx = opt_rec["best_nx"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal dt_multipler found: {optimal_dt_multipler} with n_space = {optimal_nx}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_dt_multipler = optimal_nx = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal dt_multipler found")

    optimal_param = (optimal_dt_multipler, optimal_nx)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


def find_optimal_field_order(
    profile, 
    nx, 
    dt_multipler,
    npart, 
    particle_order,
    tolerance_rmse,
    possibleOrders,
    multiplication_factor,
    max_iteration_num,
):
    """
    Grid search over field_order ∈ possibleORders for optimal limiter parameter.
    For each field_order, iterate nx until spatial convergence is achieved.

    Returns
    -------
    is_converged_optimal : bool
        Whether optimal_field_order achieved spatial convergence.
    optimal_param : (float | None, int | None)
        (optimal_field_order, optimal_nx). None if no convergent solution found.
    optimal_cost_history : list[float] | None
        Cost history for optimal_field_order corresponding to converged nx sequence.
        None if no convergent solution found.
    param_history : list
        Full parameter exploration history.
    """
    field_order_values = possibleOrders
    param_history = []
    field_order_results = []  # Save key info for each field_order (when converged)

    for field_order in field_order_values:
        field_order = int(field_order)
        print(f"\n=== Testing field_order = {field_order} ===")

        is_converged, best_nx, cost_history, one_param_history = find_convergent_nx(
            profile, dt_multipler, nx,npart, field_order, particle_order, tolerance_rmse, multiplication_factor, max_iteration_num
        )

        # Record n_space exploration trajectory for each field_order
        param_history.append(one_param_history)

        # If convergent n_space found, save to results pool
        if best_nx is not None:
            total_cost = sum(cost_history)
            field_order_results.append(
                {
                    "field_order": field_order,
                    "best_nx": best_nx,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
                }
            )
            print(f"field_order = {field_order}: Best nx = {best_nx}, Total Cost = {total_cost}")
        else:
            print(f"field_order = {field_order}: No convergent nx found")

    # Select convergent solution with minimum total cost
    if field_order_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in field_order_results]))
        opt_rec = field_order_results[min_cost_idx]

        optimal_field_order = opt_rec["field_order"]
        optimal_nx = opt_rec["best_nx"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal field_order found: {optimal_field_order} with nx = {optimal_nx}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_field_order = optimal_nx = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal field_order found")

    optimal_param = (optimal_field_order, optimal_nx)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )


def find_optimal_particle_order(
    profile, 
    nx, 
    dt_multipler,
    npart, 
    field_order,
    tolerance_rmse,
    possibleOrders,
    multiplication_factor,
    max_iteration_num,
):
    """
    Grid search over particle_order ∈ possibleOrders for optimal limiter parameter.
    For each particle_order, iterate nx until spatial convergence is achieved.

    Returns
    -------
    is_converged_optimal : bool
        Whether optimal_particle_order achieved spatial convergence.
    optimal_param : (float | None, int | None)
        (optimal_particle_order, optimal_nx). None if no convergent solution found.
    optimal_cost_history : list[float] | None
        Cost history for optimal_particle_order corresponding to converged nx sequence.
        None if no convergent solution found.
    param_history : list
        Full parameter exploration history.
    """
    particle_order_values = possibleOrders
    param_history = []
    particle_order_results = []  # Save key info for each particle_order (when converged)

    for particle_order in particle_order_values:
        particle_order = int(particle_order)
        print(f"\n=== Testing particle_order = {particle_order} ===")

        is_converged, best_nx, cost_history, one_param_history = find_convergent_nx(
            profile, dt_multipler, nx,npart, field_order, particle_order, tolerance_rmse, multiplication_factor, max_iteration_num
        )

        # Record n_space exploration trajectory for each particle_order
        param_history.append(one_param_history)

        # If convergent n_space found, save to results pool
        if best_nx is not None:
            total_cost = sum(cost_history)
            particle_order_results.append(
                {
                    "particle_order": particle_order,
                    "best_nx": best_nx,
                    "total_cost": total_cost,
                    "is_converged": is_converged,
                    "cost_history": cost_history,
                }
            )
            print(f"particle_order = {particle_order}: Best nx = {best_nx}, Total Cost = {total_cost}")
        else:
            print(f"particle_order = {particle_order}: No convergent nx found")

    # Select convergent solution with minimum total cost
    if particle_order_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in particle_order_results]))
        opt_rec = particle_order_results[min_cost_idx]

        optimal_particle_order = opt_rec["particle_order"]
        optimal_nx = opt_rec["best_nx"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = opt_rec["is_converged"]

        print(f"\nOptimal particle_order found: {optimal_particle_order} with nx = {optimal_nx}")
        print(f"Optimal cost history length: {len(optimal_cost_history)}")
    else:
        optimal_particle_order = optimal_nx = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal particle_order found")

    optimal_param = (optimal_particle_order, optimal_nx)

    return (
        is_converged_optimal,
        optimal_param,
        optimal_cost_history,
        param_history,
    )

#dt_multipler, nx, npart, field_order,particle_order,
def find_convergent_nx(profile, dt_multipler, nx, npart, field_order,particle_order, tolerance_rmse, multiplication_factor, max_iteration_num):
    """Iteratively increase nx (decrease dx) until convergence is achieved with fixed parameters."""
    nx_history = []
    cost_history = []
    param_history = []

    current_nx = int(nx)
    converged = False
    best_nx = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with nx = {current_nx}, dt_mult = {dt_multipler}, npart = {npart}, field_Order = {field_order}, particle_order = {particle_order} ")

        # Run simulation with fixed CFL
        cost_i = runEpoch(profile, current_nx, dt_multipler, npart,field_order,particle_order)
        cost_history.append(cost_i)
        nx_history.append(current_nx)
        param_history.append({"nx": current_nx, "dt_mult":  dt_multipler, "npart": npart, "field_Order" : field_order, "particle_order" : particle_order})

        # If we have previous results to compare with
        if len(nx_history) > 1:
            prev_nx = nx_history[-2]

            # Compare with previous results
            is_converged = compare_res_epoch(
                profile,
                prev_nx,
                dt_multipler,
                npart,
                field_order,
                particle_order,
                profile,
                current_nx,
                dt_multipler,
                npart,
                field_order,
                particle_order,
                tolerance_rmse
            )

            if is_converged:
                print(f"Convergence achieved between nx {prev_nx} and {current_nx}")
                best_nx = nx_history[-1]  # The finer grid that converged
                converged = True
                break
            else:
                print(f"No convergence between nx {prev_nx} and {current_nx}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal parameters for Euler 1D simulation")

    # Search mode selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["nx", "npart", "dt_multipler", "field_order","particle_order"],
        required=True,
        help="Choose which parameter to search: 'nx', 'npart', 'dt_multipler', 'field_order','particle_order'",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile configuration")

    # Controllable parameters
    parser.add_argument("--nx", type=int, default=3200, help="Initial nx (spatial discretation)")
    parser.add_argument("--npart", type=int, default=20, help="Initial npart (number pseudoparticles)")
    parser.add_argument(
        "--dt_multipler", type=float, default=0.95, help="Initial dt_multipler (temporal discretation)"
    )
    parser.add_argument(
        "--field_order", type=int, default=2, choices=[2,4,6], help="Finite-Difference order to solve field eqns"
    )
    parser.add_argument(
        "--particle_order", type=int, default=3, choices=[2,3,5], help="Particle Weighting Order"
    )

    # Tolerance parameters
    parser.add_argument("--tolerance_rmse", type=float, default=0.02, help="RMSE tolerance for convergence checking")

    # Search parameters for iterative tasks
    parser.add_argument(
        "--multiplication_factor",
        type=float,
        default=2,
        help="Factor to multiply/divide parameter values in iterative search",
    )
    parser.add_argument(
        "--max_iteration_num", type=int, default=7, help="Maximum number of iterations for iterative search"
    )

    # Search parameters for 0-shot tasks
    parser.add_argument(
        "--search_range_min", type=float, default=-1.0, help="Minimum value for 0-shot parameter search range"
    )
    parser.add_argument(
        "--search_range_max", type=float, default=1.0, help="Maximum value for 0-shot parameter search range"
    )
    parser.add_argument(
        "--search_range_slice_num", type=int, default=11, help="Number of slices for 0-shot parameter search range"
    )


    args = parser.parse_args()
    

    if args.task == "nx":
        print("\n=== Starting nx convergence search ===")
        is_converged, best_cfl, cost_history, param_history = find_convergent_nx(
            profile=args.profile,
            dt_multipler=args.dt_multipler,
            nx=args.nx,
            npart=args.npart,
            field_order=args.field_order,
            particle_order=args.particle_order,
            tolerance_rmse=args.tolerance_rmse,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_cfl is not None:
            print(f"\nRecommended nx: {best_cfl}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent nx found, total cost: {sum(cost_history)}")

    elif args.task == "npart":
        print("\n=== Starting npart parameter search ===")
        is_converged, best_npart, cost_history, param_history = find_convergent_npart(
            profile=args.profile,
            dt_multipler=args.dt_multipler,
            nx=args.nx,
            npart=args.npart,
            field_order=args.field_order,
            particle_order=args.particle_order,
            tolerance_rmse=args.tolerance_rmse,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_npart is not None:
            print(f"\nRecommended npart: {best_npart}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent npart found, total cost: {sum(cost_history)}")

    elif args.task == "dt_multipler":
        print("\n=== Starting dt_multipler parameter search ===")
        is_converged, optimal_param, optimal_cost_history, param_history = find_optimal_dt_multipler(
            profile=args.profile,
            nx=args.nx,
            npart=args.npart,
            field_order=args.field_order,
            particle_order=args.particle_order,
            tolerance_rmse=args.tolerance_rmse,
            search_range_min=args.search_range_min,
            search_range_max=args.search_range_max,
            search_range_slice_num=args.search_range_slice_num,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        optimal_dt_multipler, optimal_nx = optimal_param
        if optimal_dt_multipler is not None:
            print(f"\nRecommended k: {optimal_dt_multipler} with nx: {optimal_nx}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal k found")

    elif args.task == "field_order":
        print("\n=== Starting field_order parameter search ===")
        is_converged, optimal_param, optimal_cost_history, param_history = find_optimal_field_order(
            profile=args.profile,
            nx=args.nx,
            npart=args.npart,
            dt_multipler=args.dt_multipler,
            particle_order=args.particle_order,
            tolerance_rmse=args.tolerance_rmse,
            possibleOrders=[2,4,6],
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        optimal_field_order, optimal_nx = optimal_param
        if optimal_field_order is not None:
            print(f"\nRecommended field_order: {optimal_field_order} with nx: {optimal_nx}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal field_order found")
    elif args.task == "particle_order":
        print("\n=== Starting particle_order parameter search ===")
        is_converged, optimal_param, optimal_cost_history, param_history = find_optimal_particle_order(
            profile=args.profile,
            nx=args.nx,
            npart=args.npart,
            dt_multipler=args.dt_multipler,
            field_order=args.field_order,
            tolerance_rmse=args.tolerance_rmse,
            possibleOrders=[2,3,5],
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        optimal_particle_order, optimal_nx = optimal_param
        if optimal_particle_order is not None:
            print(f"\nRecommended particle_order: {optimal_particle_order} with nx: {optimal_nx}")
            print(f"Total cost: {sum(optimal_cost_history)}")
        else:
            print("\nNo optimal field_order found")

    else:
        print(f"\nTask type '{args.task}' is not supported.")
