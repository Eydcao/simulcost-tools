import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.hasegawa_mima_linear import run_sim_hasegawa_mima_linear, get_error_metric


def find_convergent_N(profile, N, dt, cg_atol, tolerance_rmse, multiplication_factor, max_iteration_num):
    """
    Iteratively increase N (grid resolution) until convergence is achieved.

    IMPORTANT: dt is scaled inversely with N to maintain CFL stability.
    When N increases by factor α, dt decreases by factor α (dt_new = dt_old / α).
    """
    N_history = []
    cost_history = []
    param_history = []
    error_history = []

    current_N = N
    current_dt = dt  # Track dt as it scales with N
    converged = False
    best_N = None
    best_dt = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with N = {current_N}, dt = {current_dt:.6e}, cg_atol = {cg_atol}")

        # Run simulation
        cost_i = run_sim_hasegawa_mima_linear(profile=profile, N=current_N, dt=current_dt, cg_atol=cg_atol, analytical=False)
        cost_history.append(cost_i)
        N_history.append(current_N)
        param_history.append({"N": current_N, "dt": current_dt, "cg_atol": cg_atol})

        # Get error metric
        method_suffix = "_numerical"
        sim_dir = f"sim_res/hasegawa_mima_linear/{profile}_N_{current_N}_dt_{current_dt:.2e}_cg_{cg_atol:.2e}" + method_suffix
        error = get_error_metric(sim_dir)
        error_history.append(error)

        if error is not None and error <= tolerance_rmse:
            print(f"Convergence achieved with N = {current_N}, dt = {current_dt:.6e}, error = {error:.6e}")
            best_N = current_N
            best_dt = current_dt
            converged = True
            break
        else:
            print(f"No convergence with N = {current_N}, dt = {current_dt:.6e}, error = {error}")

        # Prepare next N and dt using multiplication factor
        # dt scales inversely with N to maintain CFL stability
        next_N = int(current_N * multiplication_factor)
        next_dt = current_dt / multiplication_factor

        current_N = next_N
        current_dt = next_dt

    if converged:
        print(f"\nConvergent parameters found: N = {best_N}, dt = {best_dt:.6e}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(N_history) > 0:
            best_N = N_history[-1]
            best_dt = param_history[-1]["dt"]
            print(f"Highest tested: N = {best_N}, dt = {best_dt:.6e}")
        else:
            best_N = None
            best_dt = None

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


def find_optimal_cg_atol(profile, N, dt, tolerance_rmse, search_range_min, search_range_max, search_range_slice_num):
    """
    Iteratively search cg_atol from coarse (relaxed) to fine (strict) tolerance.
    Stops at the first convergent solution (0-shot+iterative optimization).

    Returns:
        is_converged_optimal: Whether a convergent cg_atol was found
        optimal_param: optimal_cg_atol or None
        optimal_cost_history: Cost history of all runs until convergence
        optimal_param_history: Parameter history of all runs until convergence
    """
    # Generate logarithmically spaced cg_atol values
    cg_atol_values = np.logspace(np.log10(search_range_min), np.log10(search_range_max), search_range_slice_num)
    # Start from the most relaxed (largest) cg_atol for efficiency
    cg_atol_values = cg_atol_values[::-1]

    print(f"Testing cg_atol values (coarse to fine): {cg_atol_values}")

    cg_atol_history = []
    cost_history = []
    param_history = []
    error_history = []

    converged = False
    best_cg_atol = None

    for cg_atol in cg_atol_values:
        print(f"\nRunning simulation with N = {N}, dt = {dt}, cg_atol = {cg_atol:.2e}")

        # Run simulation
        cost = run_sim_hasegawa_mima_linear(profile=profile, N=N, dt=dt, cg_atol=cg_atol, analytical=False)
        cost_history.append(cost)
        cg_atol_history.append(cg_atol)
        param_history.append({"N": N, "dt": dt, "cg_atol": cg_atol})

        # Get error metric
        method_suffix = "_numerical"
        sim_dir = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}_cg_{cg_atol:.2e}" + method_suffix
        error = get_error_metric(sim_dir)
        error_history.append(error)

        if error is not None and error <= tolerance_rmse:
            print(f"Convergence achieved with cg_atol = {cg_atol:.2e}, error = {error:.6e}")
            best_cg_atol = cg_atol
            converged = True
            break
        else:
            print(f"No convergence with cg_atol = {cg_atol:.2e}, error = {error}")

    if converged:
        print(f"\nConvergent cg_atol found: {best_cg_atol:.2e}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(cg_atol_history) > 0:
            best_cg_atol = cg_atol_history[-1]
            print(f"Finest tested cg_atol: {best_cg_atol:.2e}")
        else:
            best_cg_atol = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_cg_atol, cost_history, param_history
