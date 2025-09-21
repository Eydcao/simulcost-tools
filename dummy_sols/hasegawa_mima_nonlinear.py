import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.hasegawa_mima_nonlinear import run_sim_hasegawa_mima_nonlinear, compare_resolutions


def compare_solutions(profile, params1, params2, tolerance_rmse):
    """
    Compare two Hasegawa-Mima nonlinear simulations to check for convergence.

    Args:
        profile: Configuration profile name
        params1: Dictionary with first simulation parameters (coarse)
        params2: Dictionary with second simulation parameters (fine)
        tolerance_rmse: RMSE tolerance for convergence

    Returns:
        is_converged: Boolean indicating if solutions converged
        cost1: Cost for first simulation
        cost2: Cost for second simulation
        rmse_diff: RMSE difference between solutions
    """
    # Run both simulations
    cost1 = run_sim_hasegawa_mima_nonlinear(profile, **params1)
    cost2 = run_sim_hasegawa_mima_nonlinear(profile, **params2)

    # Get simulation directories
    dir1 = f"sim_res/hasegawa_mima_nonlinear/{profile}_N_{params1['N']}_dt_{params1['dt']:.2e}_nonlinear"
    dir2 = f"sim_res/hasegawa_mima_nonlinear/{profile}_N_{params2['N']}_dt_{params2['dt']:.2e}_nonlinear"

    # Compare resolutions
    comparison = compare_resolutions(dir1, dir2)

    if not comparison["success"]:
        return False, cost1, cost2, None

    rmse_diff = comparison["mean_l2_error"]

    # Check if error between resolutions is below tolerance
    is_converged = rmse_diff <= tolerance_rmse

    return is_converged, cost1, cost2, rmse_diff


def find_convergent_N(profile, N, dt, tolerance_rmse, multiplication_factor, max_iteration_num):
    """Iteratively increase N (grid resolution) until convergence is achieved."""
    N_history = []
    cost_history = []
    param_history = []
    error_history = []

    current_N = N
    converged = False
    best_N = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with N = {current_N}, dt = {dt}")

        # Run simulation at current resolution
        cost_i = run_sim_hasegawa_mima_nonlinear(profile=profile, N=current_N, dt=dt)
        cost_history.append(cost_i)
        N_history.append(current_N)
        param_history.append({"N": current_N, "dt": dt})

        # If this is not the first iteration, compare with previous resolution
        if i > 0:
            previous_N = N_history[-2]

            # Compare current resolution with previous resolution
            params1 = {"N": previous_N, "dt": dt}
            params2 = {"N": current_N, "dt": dt}

            is_converged, cost1, cost2, rmse_diff = compare_solutions(profile, params1, params2, tolerance_rmse)
            error_history.append(rmse_diff)

            if is_converged:
                print(f"Convergence achieved with N = {current_N}, RMSE diff = {rmse_diff:.6e}")
                best_N = current_N
                converged = True
                break
            else:
                print(f"No convergence with N = {current_N}, RMSE diff = {rmse_diff:.6e}")
        else:
            error_history.append(None)  # No comparison for first iteration

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

    # Return trajectory information
    trajectory = {
        "parameter_name": "N",
        "initial_value": N,
        "optimal_value": best_N,
        "converged": converged,
        "parameter_history": param_history,
        "cost_history": cost_history,
        "error_history": error_history,
        "N_history": N_history
    }

    return trajectory


def find_convergent_dt(profile, N, dt, tolerance_rmse, multiplication_factor, max_iteration_num):
    """Iteratively decrease dt (time step) until convergence is achieved."""
    dt_history = []
    cost_history = []
    param_history = []
    error_history = []

    current_dt = dt
    converged = False
    best_dt = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with N = {N}, dt = {current_dt}")

        # Run simulation at current time step
        cost_i = run_sim_hasegawa_mima_nonlinear(profile=profile, N=N, dt=current_dt)
        cost_history.append(cost_i)
        dt_history.append(current_dt)
        param_history.append({"N": N, "dt": current_dt})

        # If this is not the first iteration, compare with previous time step
        if i > 0:
            previous_dt = dt_history[-2]

            # Compare current time step with previous time step
            params1 = {"N": N, "dt": previous_dt}
            params2 = {"N": N, "dt": current_dt}

            is_converged, cost1, cost2, rmse_diff = compare_solutions(profile, params1, params2, tolerance_rmse)
            error_history.append(rmse_diff)

            if is_converged:
                print(f"Convergence achieved with dt = {current_dt}, RMSE diff = {rmse_diff:.6e}")
                best_dt = current_dt
                converged = True
                break
            else:
                print(f"No convergence with dt = {current_dt}, RMSE diff = {rmse_diff:.6e}")
        else:
            error_history.append(None)  # No comparison for first iteration

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

    # Return trajectory information
    trajectory = {
        "parameter_name": "dt",
        "initial_value": dt,
        "optimal_value": best_dt,
        "converged": converged,
        "parameter_history": param_history,
        "cost_history": cost_history,
        "error_history": error_history,
        "dt_history": dt_history
    }

    return trajectory


def generate_dummy_solution(profile, target_param, initial_params, tolerance_rmse, multiplication_factor, max_iteration_num):
    """
    Generate a dummy solution trajectory for parameter optimization.

    Args:
        profile: Configuration profile name
        target_param: Parameter to optimize ("N" or "dt")
        initial_params: Dictionary with initial parameter values
        tolerance_rmse: RMSE tolerance for convergence
        multiplication_factor: Factor for parameter updates
        max_iteration_num: Maximum number of iterations

    Returns:
        trajectory: Dictionary with optimization trajectory
    """
    if target_param == "N":
        return find_convergent_N(
            profile=profile,
            N=initial_params["N"],
            dt=initial_params["dt"],
            tolerance_rmse=tolerance_rmse,
            multiplication_factor=multiplication_factor,
            max_iteration_num=max_iteration_num
        )
    elif target_param == "dt":
        return find_convergent_dt(
            profile=profile,
            N=initial_params["N"],
            dt=initial_params["dt"],
            tolerance_rmse=tolerance_rmse,
            multiplication_factor=multiplication_factor,
            max_iteration_num=max_iteration_num
        )
    else:
        raise ValueError(f"Unsupported target parameter: {target_param}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy solutions for Hasegawa-Mima nonlinear solver")
    parser.add_argument("--profile", default="p1", help="Configuration profile")
    parser.add_argument("--target_param", choices=["N", "dt"], default="N", help="Target parameter to optimize")
    parser.add_argument("--N", type=int, default=64, help="Initial grid resolution")
    parser.add_argument("--dt", type=float, default=10.0, help="Initial time step")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="RMSE tolerance for convergence")
    parser.add_argument("--factor", type=float, default=2.0, help="Multiplication factor")
    parser.add_argument("--max_iter", type=int, default=3, help="Maximum iterations")

    args = parser.parse_args()

    initial_params = {"N": args.N, "dt": args.dt}

    trajectory = generate_dummy_solution(
        profile=args.profile,
        target_param=args.target_param,
        initial_params=initial_params,
        tolerance_rmse=args.tolerance,
        multiplication_factor=args.factor,
        max_iteration_num=args.max_iter
    )

    print("\nDummy solution trajectory:")
    print(f"Parameter: {trajectory['parameter_name']}")
    print(f"Initial value: {trajectory['initial_value']}")
    print(f"Optimal value: {trajectory['optimal_value']}")
    print(f"Converged: {trajectory['converged']}")
    print(f"Parameter history: {trajectory['parameter_history']}")
    print(f"Cost history: {trajectory['cost_history']}")
    print(f"Error history: {trajectory['error_history']}")