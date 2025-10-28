import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.fem2d import get_fem2d_data, compare_energies_fem2d


def find_convergent_nx(
    profile, nx, cfl, newton_v_res_tol, energy_tolerance, var_threshold, multiplication_factor, max_iteration_num
):
    """Iteratively increase nx (grid resolution) until convergence is achieved with fixed other parameters.

    Args:
        profile: Profile name (p1, p2, p3)
        nx: Initial grid resolution
        cfl: CFL number for time step calculation
        newton_v_res_tol: Newton velocity residual tolerance
        energy_tolerance: Tolerance for energy comparison
        var_threshold: Threshold for energy conservation (coefficient of variation)
        multiplication_factor: Factor to multiply nx by each iteration
        max_iteration_num: Maximum number of iterations

    Returns:
        tuple: (converged, best_nx, cost_history, param_history)
    """
    nx_history = []
    cost_history = []
    param_history = []

    current_nx = nx
    converged = False
    best_nx = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with nx = {current_nx}, cfl = {cfl}, newton_v_res_tol = {newton_v_res_tol}")

        # Run simulation and load results
        _, cost_i = get_fem2d_data(profile=profile, nx=current_nx, cfl=cfl, newton_v_res_tol=newton_v_res_tol)
        cost_history.append(cost_i)
        nx_history.append(current_nx)
        param_history.append({"nx": current_nx, "cfl": cfl, "newton_v_res_tol": newton_v_res_tol})

        # If we have previous results to compare with
        if len(nx_history) > 1:
            prev_nx = nx_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, avg_energy_diff = compare_energies_fem2d(
                profile1=profile,
                nx1=prev_nx,
                cfl1=cfl,
                newton_v_res_tol1=newton_v_res_tol,
                profile2=profile,
                nx2=current_nx,
                cfl2=cfl,
                newton_v_res_tol2=newton_v_res_tol,
                energy_tolerance=energy_tolerance,
                var_threshold=var_threshold,
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
        print(f"\nNo convergence achieved within {max_iteration_num} iterations")

    return converged, best_nx, cost_history, param_history


def find_convergent_cfl(
    profile, nx, cfl, newton_v_res_tol, energy_tolerance, var_threshold, multiplication_factor, max_iteration_num
):
    """Iteratively reduce cfl (CFL number) until convergence is achieved.

    Args:
        profile: Profile name (p1, p2, p3)
        nx: Grid resolution
        cfl: Initial CFL number
        newton_v_res_tol: Newton velocity residual tolerance
        energy_tolerance: Tolerance for energy comparison
        var_threshold: Threshold for energy conservation (coefficient of variation)
        multiplication_factor: Factor to multiply cfl by each iteration (should be < 1 to reduce cfl)
        max_iteration_num: Maximum number of iterations

    Returns:
        tuple: (converged, best_cfl, cost_history, param_history)
    """
    cfl_history = []
    cost_history = []
    param_history = []

    current_cfl = cfl
    converged = False
    best_cfl = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with nx = {nx}, cfl = {current_cfl}, newton_v_res_tol = {newton_v_res_tol}")

        # Run simulation and load results
        _, cost_i = get_fem2d_data(profile=profile, nx=nx, cfl=current_cfl, newton_v_res_tol=newton_v_res_tol)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)
        param_history.append({"nx": nx, "cfl": current_cfl, "newton_v_res_tol": newton_v_res_tol})

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            is_converged, metrics1, metrics2, avg_energy_diff = compare_energies_fem2d(
                profile1=profile,
                nx1=nx,
                cfl1=prev_cfl,
                newton_v_res_tol1=newton_v_res_tol,
                profile2=profile,
                nx2=nx,
                cfl2=current_cfl,
                newton_v_res_tol2=newton_v_res_tol,
                energy_tolerance=energy_tolerance,
                var_threshold=var_threshold,
            )

            if is_converged:
                print(f"Convergence achieved between cfl {prev_cfl} and {current_cfl}")
                best_cfl = cfl_history[-1]  # The smaller cfl that converged
                converged = True
                break
            else:
                print(f"No convergence between cfl {prev_cfl} and {current_cfl}")

        # Prepare next cfl using multiplication factor
        next_cfl = current_cfl * multiplication_factor
        current_cfl = next_cfl

    if converged:
        print(f"\nConvergent cfl found: {best_cfl}")
    else:
        print(f"\nNo convergence achieved within {max_iteration_num} iterations")

    return converged, best_cfl, cost_history, param_history


def find_optimal_newton_v_res_tol(
    profile, nx, cfl, energy_tolerance, var_threshold, search_range_min, search_range_max, search_range_slice_num
):
    """Grid search over newton_v_res_tol for optimal Newton convergence tolerance.
    Uses fixed nx and cfl values (no nested iterative search).

    Args:
        profile: Profile name (p1, p2, p3)
        nx: Fixed grid resolution
        cfl: Fixed CFL number
        energy_tolerance: Tolerance for energy comparison
        var_threshold: Threshold for energy conservation
        search_range_min: Minimum newton_v_res_tol to search
        search_range_max: Maximum newton_v_res_tol to search
        search_range_slice_num: Number of values to test in the range

    Returns:
        tuple: (converged, optimal_newton_v_res_tol, optimal_cost_history, param_history)
            - converged: True if at least one newton_v_res_tol value succeeded
            - optimal_newton_v_res_tol: The newton_v_res_tol with minimum cost
            - optimal_cost_history: Cost history (single value) for the optimal solution
            - param_history: Full exploration history for all newton_v_res_tol values
    """
    # Use logarithmic spacing for newton_v_res_tol (descending: larger to smaller tolerance)
    newton_v_res_tol_values = np.logspace(
        np.log10(search_range_max), np.log10(search_range_min), search_range_slice_num
    )

    param_history = []
    newton_results = []  # Save key info for each newton_v_res_tol

    for newton_tol in newton_v_res_tol_values:
        newton_tol = round(float(newton_tol), 6)
        print(f"\n=== Testing newton_v_res_tol = {newton_tol} ===")
        print(f"Using fixed parameters: nx = {nx}, cfl = {cfl}")

        # Run single simulation with fixed parameters
        _, cost = get_fem2d_data(profile=profile, nx=nx, cfl=cfl, newton_v_res_tol=newton_tol)

        # Record parameter combination
        param_entry = {"newton_v_res_tol": newton_tol, "nx": nx, "cfl": cfl}
        param_history.append(param_entry)

        # Record all results (fem2d doesn't have convergence criteria, all runs succeed)
        newton_results.append(
            {
                "newton_v_res_tol": newton_tol,
                "total_cost": cost,
                "cost_history": [cost],
            }
        )
        print(f"newton_v_res_tol = {newton_tol}: Total Cost = {cost}")

    # Select converged solution with minimum total cost
    if newton_results:
        min_cost_idx = int(np.argmin([r["total_cost"] for r in newton_results]))
        opt_rec = newton_results[min_cost_idx]

        optimal_newton_tol = opt_rec["newton_v_res_tol"]
        optimal_cost_history = opt_rec["cost_history"]
        is_converged_optimal = True

        print(f"\nOptimal newton_v_res_tol found: {optimal_newton_tol}")
        print(f"Optimal cost: {sum(optimal_cost_history)}")
    else:
        optimal_newton_tol = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo optimal newton_v_res_tol found")

    return is_converged_optimal, optimal_newton_tol, optimal_cost_history, param_history


if __name__ == "__main__":
    # Example usage
    profile = "p1"

    # Profile-specific initial values
    # Based on existing configs and expected mesh sizes
    initial_values = {
        "p1": {"nx": 20, "cfl": 0.5},  # cantilever: Lx=10.0, Ly=2.0
        "p2": {"nx": 70, "cfl": 0.5},  # vibration_bar: Lx=25.0, Ly=1.0
        "p3": {"nx": 20, "cfl": 0.5},  # twisting_column: Lx=2.0, Ly=10.0
    }

    initial_nx = initial_values[profile]["nx"]
    cfl = initial_values[profile]["cfl"]
    newton_v_res_tol = 0.01  # This is a 0-shot only parameter (not iteratively tuned)

    energy_tolerance = 1e-6
    var_threshold = 0.01

    # Test nx convergence
    print("=" * 60)
    print("Testing NX Convergence")
    print("=" * 60)
    converged, best_nx, cost_history, param_history = find_convergent_nx(
        profile=profile,
        nx=initial_nx,
        cfl=cfl,
        newton_v_res_tol=newton_v_res_tol,
        energy_tolerance=energy_tolerance,
        var_threshold=var_threshold,
        multiplication_factor=2,
        max_iteration_num=4,
    )

    print(f"\nNX convergence test: {converged}, best_nx: {best_nx}")
    print(f"Cost history: {cost_history}")

    # Test cfl convergence
    print("\n" + "=" * 60)
    print("Testing CFL Convergence")
    print("=" * 60)
    converged_cfl, best_cfl, cost_history_cfl, param_history_cfl = find_convergent_cfl(
        profile=profile,
        nx=initial_nx,
        cfl=cfl,
        newton_v_res_tol=newton_v_res_tol,
        energy_tolerance=energy_tolerance,
        var_threshold=var_threshold,
        multiplication_factor=0.5,  # Reduce cfl by half each iteration
        max_iteration_num=4,
    )

    print(f"\nCFL convergence test: {converged_cfl}, best_cfl: {best_cfl}")
    print(f"Cost history: {cost_history_cfl}")
