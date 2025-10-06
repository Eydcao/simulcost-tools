import argparse
import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrappers.compaction import run_sim_compaction, get_error_metric
import json


def find_convergent_nx(profile, nx, ny, tolerance_error, individual_error_tolerance, multiplication_factor, max_iteration_num):
    """Iteratively increase nx until convergence is achieved with fixed ny."""
    nx_history = []
    cost_history = []
    param_history = []

    current_nx = nx
    converged = False
    best_nx = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with nx = {current_nx}, ny = {ny}")

        # Run simulation and load results
        cost_i = run_sim_compaction(profile=profile, nx=current_nx, ny=ny)  # nx is fixed in wrapper
        cost_history.append(cost_i)
        nx_history.append(current_nx)
        param_history.append({"nx": current_nx, "ny": ny})

        # If we have previous results to compare with
        if len(nx_history) > 1:
            prev_nx = nx_history[-2]

            # Get error metrics from meta.json for both simulations
            sim_dir_prev = f"sim_res/compaction/{profile}_nx_{prev_nx}_ny_{ny}"
            sim_dir_curr = f"sim_res/compaction/{profile}_nx_{current_nx}_ny_{ny}"
            
            try:
                # Read error from meta.json
                with open(os.path.join(sim_dir_prev, "meta.json"), "r") as f:
                    meta_prev = json.load(f)
                    error_prev = meta_prev.get("error", None)
                
                with open(os.path.join(sim_dir_curr, "meta.json"), "r") as f:
                    meta_curr = json.load(f)
                    error_curr = meta_curr.get("error", None)

                if error_prev is not None and error_curr is not None:
                    # Check if error improvement is within tolerance
                    error_difference = abs(error_prev - error_curr)
                    
                    if error_difference < tolerance_error and error_curr < individual_error_tolerance:
                        print(f"Convergence achieved between nx {prev_nx} and {current_nx}")
                        print(f"Individual error tolerance met: error={error_curr:.2e} <= tolerance={individual_error_tolerance:.2e}")
                        best_nx = nx_history[-1]  # The finer grid that converged
                        converged = True
                        break
                    else:
                        print(f"No convergence between nx {prev_nx} and {current_nx} (error difference: {error_difference:.2e}, error: {error_curr:.2e})")
                else:
                    print(f"Could not extract error metrics for comparison")
            except (FileNotFoundError, KeyError) as e:
                print(f"Could not read error metrics: {e}")

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


def find_convergent_ny(profile, nx, ny, tolerance_error, individual_error_tolerance, multiplication_factor, max_iteration_num):
    """Iteratively increase ny until convergence is achieved with fixed nx."""
    ny_history = []
    cost_history = []
    param_history = []

    current_ny = ny
    converged = False
    best_ny = None

    for i in range(max_iteration_num):
        print(f"\nRunning simulation with nx = {nx}, ny = {current_ny}")

        # Run simulation and load results
        cost_i = run_sim_compaction(profile=profile, nx=nx, ny=current_ny)
        cost_history.append(cost_i)
        ny_history.append(current_ny)
        param_history.append({"nx": nx, "ny": current_ny})

        # If we have previous results to compare with
        if len(ny_history) > 1:
            prev_ny = ny_history[-2]

            # Get error metrics from meta.json for both simulations
            sim_dir_prev = f"sim_res/compaction/{profile}_nx_{nx}_ny_{prev_ny}"
            sim_dir_curr = f"sim_res/compaction/{profile}_nx_{nx}_ny_{current_ny}"
            
            try:
                # Read error from meta.json
                with open(os.path.join(sim_dir_prev, "meta.json"), "r") as f:
                    meta_prev = json.load(f)
                    error_prev = meta_prev.get("error", None)
                
                with open(os.path.join(sim_dir_curr, "meta.json"), "r") as f:
                    meta_curr = json.load(f)
                    error_curr = meta_curr.get("error", None)

                if error_prev is not None and error_curr is not None:
                    # Check if error improvement is within tolerance
                    error_difference = abs(error_prev - error_curr)
                    
                    if error_difference < tolerance_error and error_curr < individual_error_tolerance:
                        print(f"Convergence achieved between ny {prev_ny} and {current_ny}")
                        print(f"Individual error tolerance met: error={error_curr:.2e} <= tolerance={individual_error_tolerance:.2e}")
                        best_ny = ny_history[-1]  # The finer grid that converged
                        converged = True
                        break
                    else:
                        print(f"No convergence between ny {prev_ny} and {current_ny} (error difference: {error_difference:.2e}, error: {error_curr:.2e})")
                else:
                    print(f"Could not extract error metrics for comparison")
            except (FileNotFoundError, KeyError) as e:
                print(f"Could not read error metrics: {e}")

        # Prepare next ny using multiplication factor
        next_ny = int(current_ny * multiplication_factor)
        current_ny = next_ny

    if converged:
        print(f"\nConvergent ny found: {best_ny}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(ny_history) > 1:
            best_ny = ny_history[-1]
            print(f"Finest tested ny: {best_ny}")
        else:
            best_ny = None

    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")

    return bool(converged), best_ny, cost_history, param_history


def find_optimal_nx(profile, nx, ny, tolerance_error, individual_error_tolerance, search_range_min, search_range_max, search_range_slice_num):
    """Find optimal nx using grid search within specified range."""
    nx_values = np.linspace(search_range_min, search_range_max, search_range_slice_num, dtype=int)
    nx_values = np.unique(nx_values)  # Remove duplicates and sort
    
    results = []
    
    for nx_val in nx_values:
        print(f"\nTesting nx = {nx_val}")
        
        # Run simulation
        cost = run_sim_compaction(profile=profile, ny=ny)
        
        # Get error metric
        sim_dir = f"sim_res/compaction/{profile}_nx_{nx_val}_ny_{ny}"
        try:
            error = get_error_metric(sim_dir)
            results.append({"nx": nx_val, "error": error, "cost": cost})
            print(f"nx={nx_val}, error={error:.2e}, cost={cost}")
        except Exception as e:
            print(f"Failed to get error for nx={nx_val}: {e}")
            results.append({"nx": nx_val, "error": None, "cost": cost})
    
    # Find optimal nx (minimum error that meets tolerance)
    valid_results = [r for r in results if r["error"] is not None and r["error"] < individual_error_tolerance]
    
    if valid_results:
        optimal_result = min(valid_results, key=lambda x: x["error"])
        print(f"\nOptimal nx found: {optimal_result['nx']} with error {optimal_result['error']:.2e}")
        return optimal_result["nx"], results
    else:
        print("\nNo nx found meeting error tolerance")
        return None, results


def find_optimal_ny(profile, nx, ny, tolerance_error, individual_error_tolerance, search_range_min, search_range_max, search_range_slice_num):
    """Find optimal ny using grid search within specified range."""
    ny_values = np.linspace(search_range_min, search_range_max, search_range_slice_num, dtype=int)
    ny_values = np.unique(ny_values)  # Remove duplicates and sort
    
    results = []
    
    for ny_val in ny_values:
        print(f"\nTesting ny = {ny_val}")
        
        # Run simulation
        cost = run_sim_compaction(profile=profile, ny=ny_val)
        
        # Get error metric
        sim_dir = f"sim_res/compaction/{profile}_nx_{nx}_ny_{ny_val}"
        try:
            error = get_error_metric(sim_dir)
            results.append({"ny": ny_val, "error": error, "cost": cost})
            print(f"ny={ny_val}, error={error:.2e}, cost={cost}")
        except Exception as e:
            print(f"Failed to get error for ny={ny_val}: {e}")
            results.append({"ny": ny_val, "error": None, "cost": cost})
    
    # Find optimal ny (minimum error that meets tolerance)
    valid_results = [r for r in results if r["error"] is not None and r["error"] < individual_error_tolerance]
    
    if valid_results:
        optimal_result = min(valid_results, key=lambda x: x["error"])
        print(f"\nOptimal ny found: {optimal_result['ny']} with error {optimal_result['error']:.2e}")
        return optimal_result["ny"], results
    else:
        print("\nNo ny found meeting error tolerance")
        return None, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compaction dummy solution functions")
    parser.add_argument("--function", required=True, choices=["find_convergent_nx", "find_convergent_ny", "find_optimal_nx", "find_optimal_ny"],
                       help="Function to execute")
    parser.add_argument("--profile", default="p1", help="Profile name")
    parser.add_argument("--nx", type=int, default=2, help="Initial nx value")
    parser.add_argument("--ny", type=int, default=10, help="Initial ny value")
    parser.add_argument("--tolerance_error", type=float, default=0.001, help="Convergence tolerance")
    parser.add_argument("--individual_error_tolerance", type=float, default=0.01, help="Individual error tolerance")
    parser.add_argument("--multiplication_factor", type=float, default=2.0, help="Multiplication factor for iterative search")
    parser.add_argument("--max_iteration_num", type=int, default=4, help="Maximum number of iterations")
    parser.add_argument("--search_range_min", type=int, default=2, help="Minimum search range")
    parser.add_argument("--search_range_max", type=int, default=100, help="Maximum search range")
    parser.add_argument("--search_range_slice_num", type=int, default=10, help="Number of search points")

    args = parser.parse_args()

    if args.function == "find_convergent_nx":
        converged, best_nx, cost_history, param_history = find_convergent_nx(
            args.profile, args.nx, args.ny, args.tolerance_error, 
            args.individual_error_tolerance, args.multiplication_factor, args.max_iteration_num
        )
        print(f"\nFinal result: converged={converged}, best_nx={best_nx}")

    elif args.function == "find_convergent_ny":
        converged, best_ny, cost_history, param_history = find_convergent_ny(
            args.profile, args.nx, args.ny, args.tolerance_error, 
            args.individual_error_tolerance, args.multiplication_factor, args.max_iteration_num
        )
        print(f"\nFinal result: converged={converged}, best_ny={best_ny}")

    elif args.function == "find_optimal_nx":
        optimal_nx, results = find_optimal_nx(
            args.profile, args.nx, args.ny, args.tolerance_error, 
            args.individual_error_tolerance, args.search_range_min, 
            args.search_range_max, args.search_range_slice_num
        )
        print(f"\nFinal result: optimal_nx={optimal_nx}")

    elif args.function == "find_optimal_ny":
        optimal_ny, results = find_optimal_ny(
            args.profile, args.nx, args.ny, args.tolerance_error, 
            args.individual_error_tolerance, args.search_range_min, 
            args.search_range_max, args.search_range_slice_num
        )
        print(f"\nFinal result: optimal_ny={optimal_ny}")
