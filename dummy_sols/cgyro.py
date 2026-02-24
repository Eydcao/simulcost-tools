import argparse
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.cgyro import runCgyro, compare_res_cgyro, check_convergence_cgyro

# n_radial should be solvable by refinement, thus can be used to find convergence between fixed param runs
def find_convergent_nradial( 
    profile,
    n_radial,
    n_theta,
    error_tol,
    freq_tol,
    delta_t,
    n_xi,
    n_energy,
    comparison_tolerance,
    multiplication_factor,
    max_iteration_num,
):
    """Iteratively increase n_radial until convergence is achieved with fixed parameters."""
    nradial_history = []
    cost_history = []
    param_history = []

    current_nradial = int(n_radial)
    converged = False
    best_nradial = None

    for i in range(max_iteration_num):
        print(
            f"\nRunning simulation with n_radial = {current_nradial}, n_theta = {n_theta}, error_tol = {error_tol}, freq_tol = {freq_tol}, delta_t = {delta_t}, n_xi = {n_xi}, n_energy = {n_energy}"
        )

        # Run simulation with fixed params
        cost_i, _ = runCgyro(profile, current_nradial, n_theta, error_tol, freq_tol, delta_t, n_xi, n_energy)
        cost_history.append(cost_i)
        nradial_history.append(current_nradial)
        param_history.append(
            {
                "n_radial": current_nradial,
                "n_theta": n_theta,
                "error_tol": error_tol,
                "freq_tol": freq_tol,
                "delta_t": delta_t,
                "n_xi": n_xi,
                "n_energy": n_energy
            }
        )

        # If we have previous results to compare with
        if len(nradial_history) > 1:
            prev_nradial = nradial_history[-2]

            # Compare with previous results
            is_converged = compare_res_cgyro(
                profile,
                prev_nradial,
                n_theta,
                error_tol,
                freq_tol,
                delta_t,
                n_xi,
                n_energy,
                profile,
                current_nradial,
                n_theta,
                error_tol,
                freq_tol,
                delta_t,
                n_xi,
                n_energy,
                comparison_tolerance
            )

            if is_converged:
                print(f"Convergence achieved between n_radial {prev_nradial} and {current_nradial}")
                best_nradial = nradial_history[-1]  # The finer grid that converged
                converged = True
                break
            else:
                print(f"No convergence between n_radial {prev_nradial} and {current_nradial}")

        # Prepare next n_space using additive factor
        next_nradial = int(current_nradial * multiplication_factor)
        current_nradial = next_nradial

    if converged:
        print(f"\nConvergent n_radial found: {best_nradial}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(nradial_history) > 1:
            best_nradial = nradial_history[-1]
            print(f"Finest tested n_radial: {best_nradial}")
        else:
            best_nradial = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_nradial, cost_history, param_history


# n_theta should be solvable by refinement, thus can be used to find convergence between fixed param runs
def find_convergent_ntheta( 
    profile,
    n_radial,
    n_theta,
    error_tol,
    freq_tol,
    delta_t,
    n_xi,
    n_energy,
    comparison_tolerance,
    multiplication_factor,
    max_iteration_num,
):
    """Iteratively increase n_theta until convergence is achieved with fixed parameters."""
    ntheta_history = []
    cost_history = []
    param_history = []

    current_ntheta = int(n_theta)
    converged = False
    best_ntheta = None

    for i in range(max_iteration_num):
        print(
            f"\nRunning simulation with n_radial = {n_radial}, n_theta = {current_ntheta}, error_tol = {error_tol}, freq_tol = {freq_tol}, delta_t = {delta_t}, n_xi = {n_xi}, n_energy = {n_energy}"
        )

        # Run simulation with fixed params
        cost_i, _ = runCgyro(profile, n_radial, current_ntheta, error_tol, freq_tol, delta_t, n_xi, n_energy)
        cost_history.append(cost_i)
        ntheta_history.append(current_ntheta)
        param_history.append(
            {
                "n_radial": n_radial,
                "n_theta": current_ntheta,
                "error_tol": error_tol,
                "freq_tol": freq_tol,
                "delta_t": delta_t,
                "n_xi": n_xi,
                "n_energy": n_energy
            }
        )

        # If we have previous results to compare with
        if len(ntheta_history) > 1:
            prev_ntheta = ntheta_history[-2]

            # Compare with previous results
            is_converged = compare_res_cgyro(
                profile,
                n_radial,
                prev_ntheta,
                error_tol,
                freq_tol,
                delta_t,
                n_xi,
                n_energy,
                profile,
                n_radial,
                current_ntheta,
                error_tol,
                freq_tol,
                delta_t,
                n_xi,
                n_energy,
                comparison_tolerance
            )

            if is_converged:
                print(f"Convergence achieved between n_theta {prev_ntheta} and {current_ntheta}")
                best_ntheta = ntheta_history[-1]  # The finer grid that converged
                converged = True
                break
            else:
                print(f"No convergence between n_theta {prev_ntheta} and {current_ntheta}")

        # Prepare next n_space using additive factor
        next_ntheta = int(current_ntheta * multiplication_factor)
        current_ntheta = next_ntheta

    if converged:
        print(f"\nConvergent n_theta found: {best_ntheta}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(ntheta_history) > 1:
            best_ntheta = ntheta_history[-1]
            print(f"Finest tested n_theta: {best_ntheta}")
        else:
            best_ntheta = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_ntheta, cost_history, param_history

# n_xi should be solvable by refinement, thus can be used to find convergence between fixed param runs
def find_convergent_nxi( 
    profile,
    n_radial,
    n_theta,
    error_tol,
    freq_tol,
    delta_t,
    n_xi,
    n_energy,
    comparison_tolerance,
    multiplication_factor,
    max_iteration_num,
):
    """Iteratively increase n_xi until convergence is achieved with fixed parameters."""
    nxi_history = []
    cost_history = []
    param_history = []

    current_nxi = int(n_xi)
    converged = False
    best_nxi = None

    for i in range(max_iteration_num):
        print(
            f"\nRunning simulation with n_radial = {n_radial}, n_theta = {n_theta}, error_tol = {error_tol}, freq_tol = {freq_tol}, delta_t = {delta_t}, n_xi = {current_nxi}, n_energy = {n_energy}"
        )

        # Run simulation with fixed params
        cost_i, _ = runCgyro(profile, n_radial, n_theta, error_tol, freq_tol, delta_t, current_nxi, n_energy)
        cost_history.append(cost_i)
        nxi_history.append(current_nxi)
        param_history.append(
            {
                "n_radial": n_radial,
                "n_theta": n_theta,
                "error_tol": error_tol,
                "freq_tol": freq_tol,
                "delta_t": delta_t,
                "n_xi": current_nxi,
                "n_energy": n_energy
            }
        )

        # If we have previous results to compare with
        if len(nxi_history) > 1:
            prev_nxi = nxi_history[-2]

            # Compare with previous results
            is_converged = compare_res_cgyro(
                profile,
                n_radial,
                n_theta,
                error_tol,
                freq_tol,
                delta_t,
                prev_nxi,
                n_energy,
                profile,
                n_radial,
                n_theta,
                error_tol,
                freq_tol,
                delta_t,
                current_nxi,
                n_energy,
                comparison_tolerance
            )

            if is_converged:
                print(f"Convergence achieved between n_xi {prev_nxi} and {current_nxi}")
                best_nxi = nxi_history[-1]  # The finer grid that converged
                converged = True
                break
            else:
                print(f"No convergence between n_xi {prev_nxi} and {current_nxi}")

        # Prepare next n_space using additive factor
        next_nxi = int(current_nxi * multiplication_factor)
        current_nxi = next_nxi

    if converged:
        print(f"\nConvergent n_xi found: {best_nxi}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(nxi_history) > 1:
            best_nxi = nxi_history[-1]
            print(f"Finest tested n_xi: {best_nxi}")
        else:
            best_nxi = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_nxi, cost_history, param_history

# n_energy should be solvable by refinement, thus can be used to find convergence between fixed param runs
def find_convergent_nenergy( 
    profile,
    n_radial,
    n_theta,
    error_tol,
    freq_tol,
    delta_t,
    n_xi,
    n_energy,
    comparison_tolerance,
    multiplication_factor,
    max_iteration_num,
):
    """Iteratively increase n_energy until convergence is achieved with fixed parameters."""
    nenergy_history = []
    cost_history = []
    param_history = []

    current_nenergy = int(n_energy)
    converged = False
    best_nenergy = None

    for i in range(max_iteration_num):
        print(
            f"\nRunning simulation with n_radial = {n_radial}, n_theta = {n_theta}, error_tol = {error_tol}, freq_tol = {freq_tol}, delta_t = {delta_t}, n_xi = {n_xi}, n_energy = {current_nenergy}"
        )

        # Run simulation with fixed params
        cost_i, _ = runCgyro(profile, n_radial, n_theta, error_tol, freq_tol, delta_t, n_xi, current_nenergy)
        cost_history.append(cost_i)
        nenergy_history.append(current_nenergy)
        param_history.append(
            {
                "n_radial": n_radial,
                "n_theta": n_theta,
                "error_tol": error_tol,
                "freq_tol": freq_tol,
                "delta_t": delta_t,
                "n_xi": n_xi,
                "n_energy": current_nenergy
            }
        )

        # If we have previous results to compare with
        if len(nenergy_history) > 1:
            prev_nenergy = nenergy_history[-2]

            # Compare with previous results
            is_converged = compare_res_cgyro(
                profile,
                n_radial,
                n_theta,
                error_tol,
                freq_tol,
                delta_t,
                n_xi,
                prev_nenergy,
                profile,
                n_radial,
                n_theta,
                error_tol,
                freq_tol,
                delta_t,
                n_xi,
                current_nenergy,
                comparison_tolerance
            )

            if is_converged:
                print(f"Convergence achieved between n_energy {prev_nenergy} and {current_nenergy}")
                best_nenergy = nenergy_history[-1]  # The finer grid that converged
                converged = True
                break
            else:
                print(f"No convergence between n_energy {prev_nenergy} and {current_nenergy}")

        # Prepare next n_space using additive factor
        next_nenergy = int(current_nenergy * multiplication_factor)
        current_nenergy = next_nenergy

    if converged:
        print(f"\nConvergent n_energy found: {best_nenergy}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(nenergy_history) > 1:
            best_nenergy = nenergy_history[-1]
            print(f"Finest tested n_energy: {best_nenergy}")
        else:
            best_nenergy = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_nenergy, cost_history, param_history


# error_tol should be solvable by refinement, however convergence is found within each run, rather than between multiple
# In this case, we take the lowest error tolerance possible to achieve convergence, and no result comparison is necessary
def find_convergent_error_tol( 
    profile,
    n_radial,
    n_theta,
    error_tol,
    freq_tol,
    delta_t,
    multiplication_factor,
    max_iteration_num,
):
    """Iteratively increase n_theta until convergence is achieved with fixed parameters."""
    error_tol_history = []
    cost_history = []
    param_history = []

    current_error_tol = float(error_tol)
    converged = False
    best_error_tol = None

    for i in range(max_iteration_num):
        print(
            f"\nRunning simulation with n_radial = {n_radial}, n_theta = {n_theta}, error_tol = {current_error_tol}, freq_tol = {freq_tol}, delta_t = {delta_t}"
        )

        # Run simulation with fixed params
        cost_i, _ = runCgyro(profile, n_radial, n_theta, current_error_tol, freq_tol, delta_t)
        cost_history.append(cost_i)
        error_tol_history.append(current_error_tol)
        param_history.append(
            {
                "n_radial": n_radial,
                "n_theta": n_theta,
                "error_tol": current_error_tol,
                "freq_tol": freq_tol,
                "delta_t": delta_t
            }
        )

        is_converged = (check_convergence_cgyro(profile, n_radial, n_theta, current_error_tol, freq_tol, delta_t) != None)
        
        if is_converged:
            print(f"Convergence achieved at error_tol {current_error_tol}")
            best_error_tol = error_tol_history[-1]  # The finer grid that converged
            converged = True
            break
        else:
            print(f"No convergence at error_tol {current_error_tol}")

        # Prepare next n_space using multiplication factor
        next_error_tol = float(current_error_tol * multiplication_factor)
        current_error_tol = next_error_tol

    if converged:
        print(f"\nConvergent error_tol found: {best_error_tol}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(error_tol_history) > 1:
            best_error_tol = error_tol_history[-1]
            print(f"Finest tested error_tol: {best_error_tol}")
        else:
            best_error_tol = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_error_tol, cost_history, param_history

# freq_tol should be solvable by refinement, however convergence is found within each run, rather than between multiple
# In this case, we take the lowest error tolerance possible to achieve convergence, and no result comparison is necessary
def find_convergent_freq_tol( 
    profile,
    n_radial,
    n_theta,
    error_tol,
    freq_tol,
    delta_t,
    multiplication_factor,
    max_iteration_num,
):
    """Iteratively increase freq_tol until convergence is achieved with fixed parameters."""
    freq_tol_history = []
    cost_history = []
    param_history = []

    current_freq_tol = float(freq_tol)
    converged = False
    best_freq_tol = None

    for i in range(max_iteration_num):
        print(
            f"\nRunning simulation with n_radial = {n_radial}, n_theta = {n_theta}, error_tol = {error_tol}, freq_tol = {current_freq_tol}, delta_t = {delta_t}"
        )

        # Run simulation with fixed params
        cost_i, _ = runCgyro(profile, n_radial, n_theta, error_tol, current_freq_tol, delta_t)
        cost_history.append(cost_i)
        freq_tol_history.append(current_freq_tol)
        param_history.append(
            {
                "n_radial": n_radial,
                "n_theta": n_theta,
                "error_tol": error_tol,
                "freq_tol": current_freq_tol,
                "delta_t": delta_t
            }
        )

        is_converged = (check_convergence_cgyro(profile, n_radial, n_theta, error_tol, current_freq_tol, delta_t) != None)
        
        if is_converged:
            print(f"Convergence achieved at freq_tol {current_freq_tol}")
            best_freq_tol = freq_tol_history[-1]  # The finer grid that converged
            converged = True
            break
        else:
            print(f"No convergence at freq_tol {current_freq_tol}")

        # Prepare next n_space using multiplication factor
        next_freq_tol = float(current_freq_tol * multiplication_factor)
        current_freq_tol = next_freq_tol

    if converged:
        print(f"\nConvergent freq_tol found: {best_freq_tol}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(freq_tol_history) > 1:
            best_freq_tol = freq_tol_history[-1]
            print(f"Finest tested freq_tol: {best_freq_tol}")
        else:
            best_freq_tol = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_freq_tol, cost_history, param_history

# delta_t should be solvable by refinement, however convergence is found within each run, rather than between multiple
# In this case, we take the lowest error tolerance possible to achieve convergence, and no result comparison is necessary
def find_convergent_delta_t( 
    profile,
    n_radial,
    n_theta,
    error_tol,
    freq_tol,
    delta_t,
    multiplication_factor,
    max_iteration_num,
):
    """Iteratively increase freq_tol until convergence is achieved with fixed parameters."""
    delta_t_history = []
    cost_history = []
    param_history = []

    current_delta_t = float(delta_t)
    converged = False
    best_delta_t = None

    for i in range(max_iteration_num):
        print(
            f"\nRunning simulation with n_radial = {n_radial}, n_theta = {n_theta}, error_tol = {error_tol}, freq_tol = {freq_tol}, delta_t = {current_delta_t}"
        )

        # Run simulation with fixed params
        cost_i, _ = runCgyro(profile, n_radial, n_theta, error_tol, freq_tol, current_delta_t)
        cost_history.append(cost_i)
        delta_t_history.append(current_delta_t)
        param_history.append(
            {
                "n_radial": n_radial,
                "n_theta": n_theta,
                "error_tol": error_tol,
                "freq_tol": freq_tol,
                "delta_t": current_delta_t
            }
        )

        is_converged = (check_convergence_cgyro(profile, n_radial, n_theta, error_tol, freq_tol, current_delta_t) != None)
        
        if is_converged:
            print(f"Convergence achieved at delta_t {current_delta_t}")
            best_delta_t = delta_t_history[-1]  # The finer grid that converged
            converged = True
            break
        else:
            print(f"No convergence at delta_t {current_delta_t}")

        # Prepare next n_space using multiplication factor
        next_delta_t = float(current_delta_t * multiplication_factor)
        current_delta_t = next_delta_t

    if converged:
        print(f"\nConvergent delta_t found: {best_delta_t}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(delta_t_history) > 1:
            best_delta_t = delta_t_history[-1]
            print(f"Finest tested delta_t: {best_delta_t}")
        else:
            best_delta_t = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_delta_t, cost_history, param_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal parameters for CGYRO simulation")

    # Search mode selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["n_radial", "n_theta", "n_xi", "n_energy", "error_tol", "freq_tol", "delta_t"],
        required=True,
        help="Choose which parameter to search: 'n_radial', 'n_theta', 'n_xi', 'n_energy', 'error_tol', 'freq_tol', 'delta_t",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile configuration")

    # Controllable parameters
    parser.add_argument("--n_radial", type=int, default=6, help="Initial n_radial")
    parser.add_argument("--n_theta", type=int, default=16, help="Initial n_theta")
    parser.add_argument("--n_xi", type=int, default=16, help="Initial n_xi")
    parser.add_argument("--n_energy", type=int, default=8, help="Initial n_energy")
    parser.add_argument("--error_tol", type=float, default=1e-5, help="Initial error tolerance")
    parser.add_argument("--freq_tol", type=float, default=1e-4, help="Initial freq_tol")
    parser.add_argument("--delta_t", type=float, default=1e-2, help="Initial delta_t")
   
    # Tolerance parameters
    parser.add_argument("--comparison_tolerance", type=float, default=1e-3, help="Tolerance for convergence checking")

    # Search parameters for iterative tasks
    parser.add_argument(
        "--multiplication_factor",
        type=float,
        default=1.5,
        help="Factor to multiply/divide parameter values in iterative search",
    )
    parser.add_argument(
        "--max_iteration_num", type=int, default=4, help="Maximum number of iterations for iterative search"
    )

    args = parser.parse_args()

    if args.task == "n_radial":
        print("\n=== Starting n_radial convergence search ===")
        is_converged, best_nradial, cost_history, param_history = find_convergent_nradial(
            profile=args.profile,
            n_radial=args.n_radial,
            n_theta=args.n_theta,
            error_tol=args.error_tol,
            freq_tol=args.freq_tol,
            delta_t=args.delta_t,
            comparison_tolerance=args.comparison_tolerance,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_nradial is not None:
            print(f"\nRecommended n_radial: {best_nradial}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent n_radial found, total cost: {sum(cost_history)}")

    elif args.task == "n_theta":
        print("\n=== Starting n_theta parameter search ===")
        is_converged, best_ntheta, cost_history, param_history = find_convergent_ntheta(
            profile=args.profile,
            n_radial=args.n_radial,
            n_theta=args.n_theta,
            error_tol=args.error_tol,
            freq_tol=args.freq_tol,
            delta_t=args.delta_t,
            comparison_tolerance=args.comparison_tolerance,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_ntheta is not None:
            print(f"\nRecommended n_theta: {best_ntheta}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent n_theta found, total cost: {sum(cost_history)}")
    
    elif args.task == "n_xi":
        print("\n=== Starting n_xi parameter search ===")
        is_converged, best_nxi, cost_history, param_history = find_convergent_nxi(
            profile=args.profile,
            n_radial=args.n_radial,
            n_theta=args.n_theta,
            error_tol=args.error_tol,
            freq_tol=args.freq_tol,
            delta_t=args.delta_t,
            comparison_tolerance=args.comparison_tolerance,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_nxi is not None:
            print(f"\nRecommended n_xi: {best_nxi}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent n_xi found, total cost: {sum(cost_history)}")
    
    elif args.task == "n_energy":
        print("\n=== Starting n_energy parameter search ===")
        is_converged, best_nenergy, cost_history, param_history = find_convergent_nenergy(
            profile=args.profile,
            n_radial=args.n_radial,
            n_theta=args.n_theta,
            error_tol=args.error_tol,
            freq_tol=args.freq_tol,
            delta_t=args.delta_t,
            comparison_tolerance=args.comparison_tolerance,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_nenergy is not None:
            print(f"\nRecommended n_energy: {best_nenergy}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent n_energy found, total cost: {sum(cost_history)}")

    elif args.task == "error_tol":
        print("\n=== Starting error_tol parameter search ===")
        is_converged, best_error_tol, cost_history, param_history = find_convergent_error_tol(
            profile=args.profile,
            n_radial=args.n_radial,
            n_theta=args.n_theta,
            error_tol=args.error_tol,
            freq_tol=args.freq_tol,
            delta_t=args.delta_t,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_error_tol is not None:
            print(f"\nRecommended error_tol: {best_error_tol}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent error_tol found, total cost: {sum(cost_history)}")
    elif args.task == "freq_tol":
        print("\n=== Starting freq_tol parameter search ===")
        is_converged, best_freq_tol, cost_history, param_history = find_convergent_freq_tol(
            profile=args.profile,
            n_radial=args.n_radial,
            n_theta=args.n_theta,
            error_tol=args.error_tol,
            freq_tol=args.freq_tol,
            delta_t=args.delta_t,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )

        if best_freq_tol is not None:
            print(f"\nRecommended freq_tol: {best_freq_tol}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent freq_tol found, total cost: {sum(cost_history)}")
    elif args.task == "delta_t":
        print("\n=== Starting delta_t parameter search ===")
        is_converged, best_delta_t, cost_history, param_history = find_convergent_delta_t(
            profile=args.profile,
            n_radial=args.n_radial,
            n_theta=args.n_theta,
            error_tol=args.error_tol,
            freq_tol=args.freq_tol,
            delta_t=args.delta_t,
            multiplication_factor=args.multiplication_factor,
            max_iteration_num=args.max_iteration_num,
        )
        
        if best_delta_t is not None:
            print(f"\nRecommended delta_t: {best_delta_t}, total cost: {sum(cost_history)}")
        else:
            print(f"\nNo convergent delta_t found, total cost: {sum(cost_history)}")
    else:
        print(f"\nTask type '{args.task}' is not supported.")