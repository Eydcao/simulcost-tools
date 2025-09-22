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
                        search_range_slice_num, multiplication_factor, max_iteration_num):
    """
    Grid search over cg_atol in log space for optimal CG solver tolerance.
    For each cg_atol, iterate N until spatial convergence is achieved.

    Returns:
        is_converged_optimal: Whether optimal cg_atol achieved convergence
        optimal_param: (optimal_cg_atol, optimal_N) or (None, None)
        optimal_cost_history: Cost history for optimal cg_atol
        optimal_param_history: Parameter history for optimal cg_atol
    """
    # Generate logarithmically spaced cg_atol values
    cg_atol_values = np.logspace(np.log10(search_range_min), np.log10(search_range_max), search_range_slice_num)

    best_cg_atol = None
    best_N = None
    best_cost_history = None
    best_param_history = None
    converged_any = False

    for cg_atol in cg_atol_values:
        print(f"\n=== Testing cg_atol = {cg_atol:.2e} ===")

        # For each cg_atol, try to find convergent N
        is_converged, convergent_N, cost_history, param_history = find_convergent_N(
            profile=profile,
            N=N,
            dt=dt,
            cg_atol=cg_atol,
            tolerance_rmse=tolerance_rmse,
            multiplication_factor=multiplication_factor,
            max_iteration_num=max_iteration_num
        )

        if is_converged:
            print(f"Found convergent solution for cg_atol = {cg_atol:.2e}, N = {convergent_N}")
            best_cg_atol = cg_atol
            best_N = convergent_N
            best_cost_history = cost_history
            best_param_history = param_history
            converged_any = True
            break  # Take the first (most relaxed) cg_atol that works
        else:
            print(f"No convergent solution found for cg_atol = {cg_atol:.2e}")

    if converged_any:
        print(f"\nOptimal cg_atol found: {best_cg_atol:.2e} with N = {best_N}")
    else:
        print("\nNo convergent cg_atol found in the search range")

    optimal_param = (best_cg_atol, best_N) if converged_any else (None, None)

    return converged_any, optimal_param, best_cost_history, best_param_history