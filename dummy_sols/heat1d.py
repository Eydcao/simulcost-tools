import argparse
from wrappers.heat1d import run_simulation, compare_results


def find_convergent_cfl(profile, initial_cfl, initial_n_space, tolerance, max_iter):
    """Iteratively reduce CFL number until convergence is achieved."""
    cfl_history = []
    cost_history = []

    # Fix the n_space for the CFL searching
    n_space = initial_n_space

    current_cfl = initial_cfl
    converged = False
    best_cfl = None

    for i in range(max_iter):
        print(f"\nRunning simulation with CFL = {current_cfl}")

        # Run simulation and load results
        cost_i = run_simulation(profile, current_cfl, n_space)
        cost_history.append(cost_i)
        cfl_history.append(current_cfl)

        # If we have previous results to compare with
        if len(cfl_history) > 1:
            prev_cfl = cfl_history[-2]

            # Compare with previous results
            is_converged = compare_results(profile, prev_cfl, n_space, profile, current_cfl, n_space, tolerance)

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

    if not converged and len(cfl_history) > 1:
        # Check if last two simulations converged
        is_converged = compare_results(profile, cfl_history[-2], n_space, profile, cfl_history[-1], n_space, tolerance)
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

    return best_cfl, sum(cost_history)


def find_convergent_n_space(profile, initial_n_space, cfl, tolerance, max_iter):
    """Iteratively increase n_space number until convergence is achieved."""
    n_space_history = []
    cost_history = []

    # Fix the CFL for the n_space searching
    current_cfl = cfl

    current_n_space = initial_n_space
    converged = False
    best_n_space = None

    for i in range(max_iter):
        print(f"\nRunning simulation with n_space = {current_n_space}")

        # Run simulation and load results
        cost_i = run_simulation(profile, current_cfl, current_n_space)
        cost_history.append(cost_i)
        n_space_history.append(current_n_space)

        # If we have previous results to compare with
        if len(n_space_history) > 1:
            prev_n_space = n_space_history[-2]

            # Compare with previous results (with interpolation if needed)
            is_converged = compare_results(
                profile, current_cfl, prev_n_space, profile, current_cfl, current_n_space, tolerance
            )

            if is_converged:
                print(f"Convergence achieved between n_space {prev_n_space} and {current_n_space}")
                best_n_space = prev_n_space  # The coarser of the two that converged
                converged = True
                break
            else:
                print(f"No convergence between n_space {prev_n_space} and {current_n_space}")

        # Prepare next n_space (double current)
        next_n_space = current_n_space * 2
        current_n_space = next_n_space

    if not converged and len(n_space_history) > 1:
        # Check if last two simulations converged
        is_converged = compare_results(
            profile, current_cfl, n_space_history[-2], profile, current_cfl, n_space_history[-1], tolerance
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

    return best_n_space, sum(cost_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find convergent parameters for heat1d simulation")

    # Search mode selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["cfl", "n_space"],
        required=True,
        help="Choose which parameter to search: 'cfl' or 'n_space'",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile configuration")

    # CFL search parameters
    parser.add_argument(
        "--initial_cfl", type=float, default=1.0, help="Initial CFL number to start testing (for CFL search)"
    )
    parser.add_argument("--initial_n_space", type=int, default=100, help="Fixed grid number for CFL search")

    # Fixed parameters
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Tolerance for convergence checking")
    parser.add_argument("--max_iter", type=int, default=20, help="Maximum number of iterations to try")

    args = parser.parse_args()

    if args.task == "cfl":
        print("\n=== Starting CFL convergence search ===")
        best_param, total_cost = find_convergent_cfl(
            profile=args.profile,
            initial_cfl=args.initial_cfl,
            initial_n_space=args.initial_n_space,
            tolerance=args.tolerance,
            max_iter=args.max_iter,
        )
        param_name = "CFL"
    else:
        print("\n=== Starting n_space convergence search ===")
        best_param, total_cost = find_convergent_n_space(
            profile=args.profile,
            initial_n_space=args.initial_n_space,
            cfl=args.initial_cfl,
            tolerance=args.tolerance,
            max_iter=args.max_iter,
        )
        param_name = "n_space"

    if best_param is not None:
        print(f"\nRecommended {param_name}: {best_param}, the total cost is {total_cost}")
    else:
        print(f"\nNo convergent {param_name} found within the given iterations, the total cost is {total_cost}")
