import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.hasegawa_mima_linear import run_sim_hasegawa_mima_linear, get_error_metric


def compare_solutions(profile, params1, params2, tolerance_rmse):
    """
    Compare two Hasegawa-Mima simulations to check for convergence.

    Args:
        profile: Configuration profile name
        params1: Dictionary with first simulation parameters
        params2: Dictionary with second simulation parameters
        tolerance_rmse: RMSE tolerance for convergence

    Returns:
        is_converged: Boolean indicating if solutions converged
        error1: Error for first simulation
        error2: Error for second simulation
        rmse_diff: RMSE difference between solutions (not implemented yet)
    """
    # Run both simulations
    cost1 = run_sim_hasegawa_mima_linear(profile, **params1)
    cost2 = run_sim_hasegawa_mima_linear(profile, **params2)

    # Get error metrics (vs analytical solution)
    method_suffix1 = "_analytical" if params1.get('analytical', False) else "_numerical"
    method_suffix2 = "_analytical" if params2.get('analytical', False) else "_numerical"

    # Include cg_atol in path for numerical simulations
    if params1.get('analytical', False):
        dir1 = f"sim_res/hasegawa_mima_linear/{profile}_N_{params1['N']}_dt_{params1['dt']:.2e}" + method_suffix1
    else:
        dir1 = f"sim_res/hasegawa_mima_linear/{profile}_N_{params1['N']}_dt_{params1['dt']:.2e}_cg_{params1['cg_atol']:.2e}" + method_suffix1

    if params2.get('analytical', False):
        dir2 = f"sim_res/hasegawa_mima_linear/{profile}_N_{params2['N']}_dt_{params2['dt']:.2e}" + method_suffix2
    else:
        dir2 = f"sim_res/hasegawa_mima_linear/{profile}_N_{params2['N']}_dt_{params2['dt']:.2e}_cg_{params2['cg_atol']:.2e}" + method_suffix2

    error1 = get_error_metric(dir1)
    error2 = get_error_metric(dir2)

    if error1 is None or error2 is None:
        return False, error1, error2, None

    # Check if both errors are below tolerance
    is_converged = (error1 <= tolerance_rmse) and (error2 <= tolerance_rmse)

    # For now, use the difference in errors as a proxy for solution difference
    rmse_diff = abs(error1 - error2)

    return is_converged, error1, error2, rmse_diff


def find_convergent_N(profile, N, dt, cg_atol, tolerance_rmse, multiplication_factor, max_iteration_num):
    """Iteratively increase N (grid resolution) until convergence is achieved."""
    N_history = []
    cost_history = []
    param_history = []
    error_history = []

    current_N = N
    converged = False
    best_N = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with N = {current_N}, dt = {dt}, cg_atol = {cg_atol}")

        # Run simulation
        cost_i = run_sim_hasegawa_mima_linear(profile=profile, N=current_N, dt=dt, cg_atol=cg_atol, analytical=False)
        cost_history.append(cost_i)
        N_history.append(current_N)
        param_history.append({"N": current_N, "dt": dt, "cg_atol": cg_atol})

        # Get error metric
        method_suffix = "_numerical"
        sim_dir = f"sim_res/hasegawa_mima_linear/{profile}_N_{current_N}_dt_{dt:.2e}_cg_{cg_atol:.2e}" + method_suffix
        error = get_error_metric(sim_dir)
        error_history.append(error)

        if error is not None and error <= tolerance_rmse:
            print(f"Convergence achieved with N = {current_N}, error = {error:.6e}")
            best_N = current_N
            converged = True
            break
        else:
            print(f"No convergence with N = {current_N}, error = {error}")

        # Prepare next N using multiplication factor
        next_N = int(current_N * multiplication_factor)
        current_N = next_N

    if converged:
        print(f"\nConvergent N found: {best_N}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(N_history) > 0:
            best_N = N_history[-1]
            print(f"Highest tested N: {best_N}")
        else:
            best_N = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_N, cost_history, param_history


def find_convergent_dt(profile, N, dt, cg_atol, tolerance_rmse, multiplication_factor, max_iteration_num):
    """Iteratively reduce dt (time step) until convergence is achieved."""
    dt_history = []
    cost_history = []
    param_history = []
    error_history = []

    current_dt = dt
    converged = False
    best_dt = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with N = {N}, dt = {current_dt}, cg_atol = {cg_atol}")

        # Run simulation
        cost_i = run_sim_hasegawa_mima_linear(profile=profile, N=N, dt=current_dt, cg_atol=cg_atol, analytical=False)
        cost_history.append(cost_i)
        dt_history.append(current_dt)
        param_history.append({"N": N, "dt": current_dt, "cg_atol": cg_atol})

        # Get error metric
        method_suffix = "_numerical"
        sim_dir = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{current_dt:.2e}_cg_{cg_atol:.2e}" + method_suffix
        error = get_error_metric(sim_dir)
        error_history.append(error)

        if error is not None and error <= tolerance_rmse:
            print(f"Convergence achieved with dt = {current_dt}, error = {error:.6e}")
            best_dt = current_dt
            converged = True
            break
        else:
            print(f"No convergence with dt = {current_dt}, error = {error}")

        # Prepare next dt using multiplication factor
        next_dt = current_dt * multiplication_factor
        current_dt = next_dt

    if converged:
        print(f"\nConvergent dt found: {best_dt}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(dt_history) > 0:
            best_dt = dt_history[-1]
            print(f"Smallest tested dt: {best_dt}")
        else:
            best_dt = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_dt, cost_history, param_history

def find_optimal_cg_atol(profile, N, dt, tolerance_rmse, search_range_min, search_range_max,
                        search_range_slice_num):
    """
    Grid search over cg_atol in log space for optimal CG solver tolerance.
    Uses fixed N and dt parameters (0-shot optimization).

    Returns:
        is_converged_optimal: Whether optimal cg_atol achieved convergence
        optimal_param: optimal_cg_atol or None
        optimal_cost_history: Cost history for optimal cg_atol
        optimal_param_history: Parameter history for optimal cg_atol
    """
    # Generate logarithmically spaced cg_atol values
    cg_atol_values = np.logspace(np.log10(search_range_min), np.log10(search_range_max), search_range_slice_num)
    # Start from the most relaxed (largest) cg_atol for efficiency
    cg_atol_values = cg_atol_values[::-1]

    param_history = []
    cg_atol_results = []  # Save key info for each cg_atol when converged

    for cg_atol in cg_atol_values:
        print(f"\n=== Testing cg_atol = {cg_atol:.2e} ===")

        # Run single simulation with fixed parameters
        print(f"Running simulation with N = {N}, dt = {dt}, cg_atol = {cg_atol}")

        # Run simulation
        cost = run_sim_hasegawa_mima_linear(profile=profile, N=N, dt=dt, cg_atol=cg_atol, analytical=False)

        # Record parameter combination
        param_entry = {"N": N, "dt": dt, "cg_atol": cg_atol}
        param_history.append(param_entry)

        # Get error metric
        method_suffix = "_numerical"
        sim_dir = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}_cg_{cg_atol:.2e}" + method_suffix
        error = get_error_metric(sim_dir)

        if error is not None and error <= tolerance_rmse:
            print(f"Convergence achieved with cg_atol = {cg_atol:.2e}, error = {error:.6e}")

            # Store this result
            cg_atol_results.append({
                "cg_atol": cg_atol,
                "cost": cost,
                "error": error,
                "converged": True
            })
        else:
            print(f"No convergence with cg_atol = {cg_atol:.2e}, error = {error}")

            # Store failed result
            cg_atol_results.append({
                "cg_atol": cg_atol,
                "cost": cost,
                "error": error,
                "converged": False
            })

    # Select convergent solution with minimum cost
    converged_results = [r for r in cg_atol_results if r["converged"]]

    if converged_results:
        # Choose the solution with minimum cost among converged ones
        min_cost_idx = int(np.argmin([r["cost"] for r in converged_results]))
        opt_rec = converged_results[min_cost_idx]

        optimal_cg_atol = opt_rec["cg_atol"]
        optimal_cost_history = [opt_rec["cost"]]  # Single cost since no iteration
        is_converged_optimal = True

        print(f"\nOptimal cg_atol found: {optimal_cg_atol:.2e}")
        print(f"Cost: {opt_rec['cost']}, Error: {opt_rec['error']:.2e}")
    else:
        optimal_cg_atol = None
        optimal_cost_history = None
        is_converged_optimal = False
        print("\nNo convergent cg_atol found in the search range")

    return is_converged_optimal, optimal_cg_atol, optimal_cost_history, param_history


def main():
    profile = "p1"
    N = 128
    dt = 1.0e1
    tolerance_rmse = 0.001
    search_range_min = 1e-6
    search_range_max = 1e-2
    search_range_slice_num = 5

    print("Starting test of find_optimal_cg_atol with parameters:")
    print(f"profile={profile}, N={N}, dt={dt:.2e}, tolerance_rmse={tolerance_rmse}")
    print(f"cg_atol range: {search_range_min:.0e} -> {search_range_max:.0e}, slices={search_range_slice_num}")
    print("Fixed N and dt (0-shot optimization for cg_atol)")

    converged, optimal_cg_atol, cost_history, param_history = find_optimal_cg_atol(
        profile=profile,
        N=N,
        dt=dt,
        tolerance_rmse=tolerance_rmse,
        search_range_min=search_range_min,
        search_range_max=search_range_max,
        search_range_slice_num=search_range_slice_num
    )

    print("\nTest result:")
    print(f"converged: {converged}")
    print(f"optimal_cg_atol: {optimal_cg_atol}")
    print(f"cost_history: {cost_history}")
    print(f"param_history: {param_history}")


if __name__ == "__main__":
    main()
