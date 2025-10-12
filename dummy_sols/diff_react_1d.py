import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrappers import *


def find_convergent_cfl(profile, initial_cfl, initial_n_space, tolerance, max_iter, multiplication_factor, 
                       reaction_type="fisher", allee_threshold=None):
    """Iteratively reduce CFL number until convergence is achieved."""
    cfl_history = []
    cost_history = []
    param_history = []

    # Fix the n_space and other parameters for the CFL searching
    n_space = initial_n_space
    tol = 1e-9
    min_step = 1e-3
    initial_step_guess = 1.0

    current_cfl = initial_cfl
    converged = False
    best_cfl = None

    for i in range(max_iter):
        print(f"\nRunning simulation with CFL = {current_cfl}")

        # Run simulation and load results
        cost_i = run_sim_diff_react_1d(profile, n_space, current_cfl, tol, min_step, initial_step_guess, 
                                      reaction_type, allee_threshold)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)
        param_history.append({"cfl": current_cfl, "n_space": n_space, "tol": tol, "min_step": min_step, "initial_step_guess": initial_step_guess})

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            is_converged, _, _, _ = compare_res_diff_react_1d(
                profile, n_space, prev_cfl, tol, min_step, initial_step_guess,
                profile, n_space, current_cfl, tol, min_step, initial_step_guess,
                tolerance, reaction_type, reaction_type, allee_threshold, allee_threshold
            )

            if is_converged:
                print(f"Convergence achieved between CFL {prev_cfl} and {current_cfl}")
                best_cfl = cfl_history[-1]  # The larger of the two CFLs that converged
                converged = True
                break
            else:
                print(f"No convergence between CFL {prev_cfl} and {current_cfl}")

        # Prepare next CFL using multiplication factor
        from solvers.utils import format_param_for_path
        current_cfl = float(format_param_for_path(current_cfl * multiplication_factor))

    if not converged and len(cfl_history) > 1:
        # Check if last two simulations converged
        is_converged, _, _, _ = compare_res_diff_react_1d(
            profile, n_space, cfl_history[-2], tol, min_step, initial_step_guess,
            profile, n_space, cfl_history[-1], tol, min_step, initial_step_guess,
            tolerance, reaction_type, reaction_type, allee_threshold, allee_threshold
        )
        if is_converged:
            best_cfl = cfl_history[-2]
            converged = True

    if converged:
        print(f"\nConvergent CFL number found: {best_cfl}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(cfl_history) > 1:
            best_cfl = cfl_history[-1]
            print(f"Smallest tested CFL: {best_cfl}")
        else:
            best_cfl = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(is_converged), best_cfl, cost_history, param_history


def find_convergent_n_space(profile, initial_n_space, cfl, tolerance, max_iter, multiplication_factor,
                           reaction_type="fisher", allee_threshold=None):
    """Iteratively increase n_space number until convergence is achieved."""
    n_space_history = []
    cost_history = []
    param_history = []

    # Fix the CFL and other parameters for the n_space searching
    current_cfl = cfl
    tol = 1e-9
    min_step = 1e-3
    initial_step_guess = 1.0

    current_n_space = initial_n_space
    converged = False
    best_n_space = None

    for i in range(max_iter):
        print(f"\nRunning simulation with n_space = {current_n_space}")

        # Run simulation and load results
        cost_i = run_sim_diff_react_1d(profile, current_n_space, current_cfl, tol, min_step, initial_step_guess,
                                      reaction_type, allee_threshold)
        cost_history.append(cost_i)
        n_space_history.append(current_n_space)
        param_history.append({"cfl": current_cfl, "n_space": current_n_space, "tol": tol, "min_step": min_step, "initial_step_guess": initial_step_guess})

        # If we have previous results to compare with
        if len(n_space_history) > 1:
            prev_n_space = n_space_history[-2]

            # Compare with previous results (with interpolation if needed)
            is_converged, _, _, _ = compare_res_diff_react_1d(
                profile, prev_n_space, current_cfl, tol, min_step, initial_step_guess,
                profile, current_n_space, current_cfl, tol, min_step, initial_step_guess,
                tolerance, reaction_type, reaction_type, allee_threshold, allee_threshold
            )

            if is_converged:
                print(f"Convergence achieved between n_space {prev_n_space} and {current_n_space}")
                best_n_space = n_space_history[-1]  # The coarser of the two that converged
                converged = True
                break
            else:
                print(f"No convergence between n_space {prev_n_space} and {current_n_space}")

        # Prepare next n_space using multiplication factor
        next_n_space = current_n_space * multiplication_factor
        current_n_space = next_n_space

    if not converged and len(n_space_history) > 1:
        # Check if last two simulations converged
        is_converged, _, _, _ = compare_res_diff_react_1d(
            profile, n_space_history[-2], current_cfl, tol, min_step, initial_step_guess,
            profile, n_space_history[-1], current_cfl, tol, min_step, initial_step_guess,
            tolerance, reaction_type, reaction_type, allee_threshold, allee_threshold
        )
        if is_converged:
            best_n_space = n_space_history[-2]
            converged = True

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

    return bool(is_converged), best_n_space, cost_history, param_history


def find_convergent_tolerance(profile, initial_tol, n_space, cfl, tolerance, max_iter, multiplication_factor,
                            reaction_type="fisher", allee_threshold=None):
    """Iteratively tighten tolerance until convergence is achieved."""
    tol_history = []
    cost_history = []
    param_history = []

    # Fix other parameters for the tolerance searching
    min_step = 1e-3
    initial_step_guess = 1.0

    current_tol = initial_tol
    converged = False
    best_tol = None

    for i in range(max_iter):
        print(f"\nRunning simulation with tolerance = {current_tol}")

        # Run simulation and load results
        cost_i = run_sim_diff_react_1d(profile, n_space, cfl, current_tol, min_step, initial_step_guess,
                                      reaction_type, allee_threshold)
        cost_history.append(cost_i)
        tol_history.append(current_tol)
        param_history.append({"cfl": cfl, "n_space": n_space, "tol": current_tol, "min_step": min_step, "initial_step_guess": initial_step_guess})

        # If we have previous results to compare with
        if len(tol_history) > 1:
            prev_tol = tol_history[-2]

            # Compare with previous results
            is_converged, _, _, _ = compare_res_diff_react_1d(
                profile, n_space, cfl, prev_tol, min_step, initial_step_guess,
                profile, n_space, cfl, current_tol, min_step, initial_step_guess,
                tolerance, reaction_type, reaction_type, allee_threshold, allee_threshold
            )

            if is_converged:
                print(f"Convergence achieved between tolerance {prev_tol} and {current_tol}")
                best_tol = tol_history[-1]  # The looser of the two tolerances that converged
                converged = True
                break
            else:
                print(f"No convergence between tolerance {prev_tol} and {current_tol}")

        # Prepare next tolerance using multiplication factor
        from solvers.utils import format_param_for_path
        current_tol = float(format_param_for_path(current_tol * multiplication_factor))

    if not converged and len(tol_history) > 1:
        # Check if last two simulations converged
        is_converged, _, _, _ = compare_res_diff_react_1d(
            profile, n_space, cfl, tol_history[-2], min_step, initial_step_guess,
            profile, n_space, cfl, tol_history[-1], min_step, initial_step_guess,
            tolerance, reaction_type, reaction_type, allee_threshold, allee_threshold
        )
        if is_converged:
            best_tol = tol_history[-2]
            converged = True

    if converged:
        print(f"\nConvergent tolerance found: {best_tol}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(tol_history) > 1:
            best_tol = tol_history[-1]
            print(f"Tightest tested tolerance: {best_tol}")
        else:
            best_tol = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(is_converged), best_tol, cost_history, param_history


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Find convergent parameters for diff_react_1d")
    parser.add_argument("--profile", default="p1", help="Profile name")
    parser.add_argument("--reaction_type", default="fisher", choices=["fisher", "allee", "cubic"], help="Reaction type")
    parser.add_argument("--allee_threshold", type=float, default=0.3, help="Allee threshold (for allee reaction)")
    parser.add_argument("--initial_cfl", type=float, default=1.0, help="Initial CFL number")
    parser.add_argument("--initial_n_space", type=int, default=512, help="Initial spatial resolution")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Convergence tolerance")
    parser.add_argument("--max_iter", type=int, default=5, help="Maximum iterations")
    parser.add_argument("--multiplication_factor", type=float, default=0.5, help="Multiplication factor for parameter adjustment")
    
    args = parser.parse_args()

    print(f"Finding convergent parameters for {args.reaction_type} reaction...")
    
    # Find convergent CFL
    print("\n=== Finding convergent CFL ===")
    cfl_converged, best_cfl, cfl_costs, cfl_params = find_convergent_cfl(
        args.profile, args.initial_cfl, args.initial_n_space, args.tolerance, 
        args.max_iter, args.multiplication_factor, args.reaction_type, args.allee_threshold
    )
    
    # Find convergent n_space
    print("\n=== Finding convergent n_space ===")
    nspace_converged, best_nspace, nspace_costs, nspace_params = find_convergent_n_space(
        args.profile, args.initial_n_space, best_cfl or args.initial_cfl, args.tolerance,
        args.max_iter, 2.0, args.reaction_type, args.allee_threshold
    )
    
    # Find convergent tolerance
    print("\n=== Finding convergent tolerance ===")
    tol_converged, best_tol, tol_costs, tol_params = find_convergent_tolerance(
        args.profile, 1e-6, best_nspace or args.initial_n_space, best_cfl or args.initial_cfl,
        args.tolerance, args.max_iter, 0.1, args.reaction_type, args.allee_threshold
    )
    
    print(f"\n=== Summary ===")
    print(f"CFL convergence: {cfl_converged}, best CFL: {best_cfl}")
    print(f"n_space convergence: {nspace_converged}, best n_space: {best_nspace}")
    print(f"Tolerance convergence: {tol_converged}, best tolerance: {best_tol}")
    print(f"Total cost: {sum(cfl_costs) + sum(nspace_costs) + sum(tol_costs)}")


def find_convergent_min_step(profile, initial_min_step, n_space, cfl, tolerance, max_iter, multiplication_factor, 
                            reaction_type="fisher", allee_threshold=None):
    """
    Find convergent min_step parameter for DiffReact1D solver.
    
    Args:
        profile: Profile name (p1, p2, p3)
        initial_min_step: Initial min_step value to test
        n_space: Number of spatial grid points
        cfl: CFL number
        tolerance: RMSE tolerance for convergence
        max_iter: Maximum number of iterations
        multiplication_factor: Factor to multiply min_step by each iteration
        reaction_type: Type of reaction term
        allee_threshold: Threshold for Allee effect (if applicable)
    
    Returns:
        tuple: (is_converged, best_min_step, cost_history, min_step_history)
    """
    print(f"\n=== Finding convergent min_step for profile {profile} ===")
    print(f"Initial min_step: {initial_min_step}")
    print(f"Other parameters: n_space={n_space}, cfl={cfl}, tolerance={tolerance}")
    
    min_step_history = [initial_min_step]
    cost_history = []
    converged = False
    best_min_step = None
    
    # Fixed parameters
    tol = 1e-9
    initial_step_guess = 1.0
    
    for i in range(max_iter):
        current_min_step = min_step_history[-1]
        print(f"\nIteration {i+1}: Testing min_step = {current_min_step}")
        
        # Run simulation with current min_step
        cost = run_sim_diff_react_1d(
            profile=profile,
            n_space=n_space,
            cfl=cfl,
            tol=tol,
            min_step=current_min_step,
            initial_step_guess=initial_step_guess,
            reaction_type=reaction_type,
            allee_threshold=allee_threshold
        )
        cost_history.append(cost)
        print(f"Simulation completed with cost: {cost}")
        
        # Check convergence with previous iteration
        if i > 0:
            is_converged, _, _, _ = compare_res_diff_react_1d(
                profile, n_space, cfl, tol, min_step_history[-2], initial_step_guess,
                profile, n_space, cfl, tol, current_min_step, initial_step_guess,
                tolerance, reaction_type, reaction_type, allee_threshold, allee_threshold
            )
            if is_converged:
                best_min_step = min_step_history[-2]
                converged = True
                print(f"Convergence found between min_step {min_step_history[-2]} and {current_min_step}")
                break
        else:
            print(f"First iteration with min_step {current_min_step}, no previous value to compare")
        
        # Prepare next min_step using multiplication factor
        from solvers.utils import format_param_for_path
        current_min_step = float(format_param_for_path(current_min_step * multiplication_factor))
        min_step_history.append(current_min_step)
    
    if not converged and len(min_step_history) > 1:
        # Check if last two simulations converged
        is_converged, _, _, _ = compare_res_diff_react_1d(
            profile, n_space, cfl, tol, min_step_history[-2], initial_step_guess,
            profile, n_space, cfl, tol, min_step_history[-1], initial_step_guess,
            tolerance, reaction_type, reaction_type, allee_threshold, allee_threshold
        )
        if is_converged:
            best_min_step = min_step_history[-2]
            converged = True
    
    if converged:
        print(f"\nConvergent min_step found: {best_min_step}")
    else:
        print(f"\nNo convergent min_step found in {max_iter} iterations")
    
    return converged, best_min_step, cost_history, min_step_history


def find_convergent_initial_step_guess(profile, initial_step_guess, n_space, cfl, tolerance, max_iter, multiplication_factor,
                                     reaction_type="fisher", allee_threshold=None):
    """
    Find convergent initial_step_guess parameter for DiffReact1D solver.
    
    Args:
        profile: Profile name (p1, p2, p3)
        initial_step_guess: Initial step guess value to test
        n_space: Number of spatial grid points
        cfl: CFL number
        tolerance: RMSE tolerance for convergence
        max_iter: Maximum number of iterations
        multiplication_factor: Factor to multiply initial_step_guess by each iteration
        reaction_type: Type of reaction term
        allee_threshold: Threshold for Allee effect (if applicable)
    
    Returns:
        tuple: (is_converged, best_initial_step_guess, cost_history, initial_step_guess_history)
    """
    print(f"\n=== Finding convergent initial_step_guess for profile {profile} ===")
    print(f"Initial initial_step_guess: {initial_step_guess}")
    print(f"Other parameters: n_space={n_space}, cfl={cfl}, tolerance={tolerance}")
    
    initial_step_guess_history = [initial_step_guess]
    cost_history = []
    converged = False
    best_initial_step_guess = None
    
    # Fixed parameters
    tol = 1e-9
    min_step = 1e-3
    
    for i in range(max_iter):
        current_initial_step_guess = initial_step_guess_history[-1]
        print(f"\nIteration {i+1}: Testing initial_step_guess = {current_initial_step_guess}")
        
        # Run simulation with current initial_step_guess
        cost = run_sim_diff_react_1d(
            profile=profile,
            n_space=n_space,
            cfl=cfl,
            tol=tol,
            min_step=min_step,
            initial_step_guess=current_initial_step_guess,
            reaction_type=reaction_type,
            allee_threshold=allee_threshold
        )
        cost_history.append(cost)
        print(f"Simulation completed with cost: {cost}")
        
        # Check convergence with previous iteration
        if i > 0:
            is_converged, _, _, _ = compare_res_diff_react_1d(
                profile, n_space, cfl, tol, min_step, initial_step_guess_history[-2],
                profile, n_space, cfl, tol, min_step, current_initial_step_guess,
                tolerance, reaction_type, reaction_type, allee_threshold, allee_threshold
            )
            if is_converged:
                best_initial_step_guess = initial_step_guess_history[-2]
                converged = True
                print(f"Convergence found between initial_step_guess {initial_step_guess_history[-2]} and {current_initial_step_guess}")
                break
        else:
            print(f"First iteration with initial_step_guess {current_initial_step_guess}, no previous value to compare")
        
        # Prepare next initial_step_guess using multiplication factor
        from solvers.utils import format_param_for_path
        current_initial_step_guess = float(format_param_for_path(current_initial_step_guess * multiplication_factor))
        initial_step_guess_history.append(current_initial_step_guess)
    
    if not converged and len(initial_step_guess_history) > 1:
        # Check if last two simulations converged
        is_converged, _, _, _ = compare_res_diff_react_1d(
            profile, n_space, cfl, tol, min_step, initial_step_guess_history[-2],
            profile, n_space, cfl, tol, min_step, initial_step_guess_history[-1],
            tolerance, reaction_type, reaction_type, allee_threshold, allee_threshold
        )
        if is_converged:
            best_initial_step_guess = initial_step_guess_history[-2]
            converged = True
    
    if converged:
        print(f"\nConvergent initial_step_guess found: {best_initial_step_guess}")
    else:
        print(f"\nNo convergent initial_step_guess found in {max_iter} iterations")
    
    return converged, best_initial_step_guess, cost_history, initial_step_guess_history
