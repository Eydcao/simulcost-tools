import itertools
import os
import sys
import json
import time
from collections import defaultdict

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dummy_sols.euler_2d import find_convergent_n_grid_x, find_convergent_cfl, grid_search_cg_tolerance
from checkouts.config_utils import load_config, build_target_configs


def save_datasets(successful_tasks, failed_tasks, output_dir):
    """Save successful and failed tasks as separate JSON datasets in subfolders"""
    # Create subfolders for successful and failed tasks
    success_dir = os.path.join(output_dir, "euler_2d", "successful")
    failed_dir = os.path.join(output_dir, "euler_2d", "failed")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    # Save successful tasks
    success_file = os.path.join(success_dir, "tasks.json")
    with open(success_file, "w") as f:
        json.dump({
            "metadata": {
                "solver": "euler_2d",
                "description": "Successfully converged parameter optimization tasks",
                "total_tasks": len(successful_tasks)
            },
            "tasks": successful_tasks
        }, f, indent=2)

    # Save failed tasks
    failed_file = os.path.join(failed_dir, "tasks.json")
    with open(failed_file, "w") as f:
        json.dump({
            "metadata": {
                "solver": "euler_2d",
                "description": "Failed to converge parameter optimization tasks",
                "total_tasks": len(failed_tasks)
            },
            "tasks": failed_tasks
        }, f, indent=2)

    print(f"✅ Saved {len(successful_tasks)} successful tasks to {success_file}")
    print(f"❌ Saved {len(failed_tasks)} failed tasks to {failed_file}")

    return success_file, failed_file


def run_checkout(config_path, output_dir="dataset", profiles_to_test=None):
    """
    Run the complete checkout process for Euler 2D solver.

    Args:
        config_path: Path to euler_2d.yaml config file
        output_dir: Directory to save results
        profiles_to_test: List of profile names to test (e.g., ["p1", "p2"]). If None, tests all active profiles.
    """
    # Load configuration
    config = load_config(config_path)

    # Get active profiles
    active_profiles = config["profiles"]["active_profiles"]
    if profiles_to_test:
        active_profiles = [p for p in active_profiles if p in profiles_to_test]

    print(f"{'='*80}")
    print(f"Euler 2D Solver Checkout")
    print(f"{'='*80}")
    print(f"Active profiles: {active_profiles}")
    print(f"Precision levels: {list(config['precision_levels'].keys())}")
    print(f"Target parameters: {list(config['target_parameters'].keys())}")
    print(f"{'='*80}\n")

    # Build all target configurations (parameter-level configs)
    target_param_configs = build_target_configs(config)

    # Generate all task combinations: profiles × precision levels × target parameters × non-target combinations
    all_tasks = []
    for profile in active_profiles:
        for precision_level in config["precision_levels"].keys():
            for target_param, param_config in target_param_configs.items():
                # Get all non-target parameter names and their value lists
                non_target_params = param_config["non_target_parameters"]
                param_names = list(non_target_params.keys())
                param_values = [non_target_params[name] for name in param_names]

                # Generate all combinations using itertools.product
                for combination in itertools.product(*param_values):
                    # Build non-target parameters dictionary for this combination
                    task_non_target_params = dict(zip(param_names, combination))

                    task = {
                        "profile": profile,
                        "precision": precision_level,
                        "tolerance_rmse": config["precision_levels"][precision_level]["tolerance_rmse"],
                        "target_parameter": target_param,
                        "non_target_params_combo": task_non_target_params,  # Store the specific combination
                        **param_config  # Unpack search parameters (will be overridden by specific combo below)
                    }
                    all_tasks.append(task)

    print(f"Total tasks to execute: {len(all_tasks)}\n")

    # Storage for results
    successful_tasks = []
    failed_tasks = []

    # Statistics tracking
    statistics = {
        "convergence_by_precision": defaultdict(lambda: {"total": 0, "converged": 0}),
        "convergence_by_profile": defaultdict(lambda: {"total": 0, "converged": 0}),
        "total_cost": 0,
        "total_time": 0,
        "tasks_completed": 0,
    }

    start_time_total = time.time()

    # Execute each task
    for idx, task_config in enumerate(all_tasks, 1):
        profile = task_config["profile"]
        precision = task_config["precision"]
        target_param = task_config["target_parameter"]
        tolerance_rmse = task_config["tolerance_rmse"]

        # Extract testcase and end_frame from profile config
        # Load the profile config to get the correct end_frame
        import yaml
        profile_config_path = f"run_configs/euler_2d/{profile}.yaml"
        with open(profile_config_path, "r") as f:
            profile_config = yaml.safe_load(f)

        testcase = profile_config["testcase"]
        end_frame = profile_config["end_frame"]

        print(f"\n{'='*80}")
        print(f"Task {idx}/{len(all_tasks)}")
        print(f"Profile: {profile} (testcase={testcase})")
        print(f"Precision: {precision} (tolerance_rmse={tolerance_rmse})")
        print(f"Target: {target_param}")
        print(f"{'='*80}")

        # Update statistics
        statistics["convergence_by_precision"][precision]["total"] += 1
        statistics["convergence_by_profile"][profile]["total"] += 1

        task_start_time = time.time()

        try:
            # Get the specific non-target parameter combination for this task
            non_target_params = task_config.get("non_target_params_combo", {})

            # Run search based on target parameter
            if target_param == "n_grid_x":
                is_converged, best_n_grid_x, cost_history, param_history = find_convergent_n_grid_x(
                    profile=profile,
                    testcase=testcase,
                    n_grid_x=task_config["initial_value"],
                    cfl=non_target_params.get("cfl", 0.5),
                    cg_tolerance=non_target_params.get("cg_tolerance", 1.0e-7),
                    start_frame=0,
                    end_frame=end_frame,
                    tolerance_rmse=tolerance_rmse,
                    multiplication_factor=task_config["multiplication_factor"],
                    max_iteration_num=task_config["max_iteration_num"]
                )
                optimal_params = {"n_grid_x": best_n_grid_x}

            elif target_param == "cfl":
                is_converged, best_cfl, cost_history, param_history = find_convergent_cfl(
                    profile=profile,
                    testcase=testcase,
                    n_grid_x=non_target_params.get("n_grid_x", 64),
                    cfl=task_config["initial_value"],
                    cg_tolerance=non_target_params.get("cg_tolerance", 1.0e-7),
                    start_frame=0,
                    end_frame=end_frame,
                    tolerance_rmse=tolerance_rmse,
                    division_factor=task_config.get("division_factor", 2.0),
                    max_iteration_num=task_config["max_iteration_num"]
                )
                optimal_params = {"cfl": best_cfl}

            elif target_param == "cg_tolerance":
                cg_tolerance_values = task_config.get("search_values", [1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5])
                is_converged, best_cg_tolerance, cost_history, param_history = grid_search_cg_tolerance(
                    profile=profile,
                    testcase=testcase,
                    n_grid_x=non_target_params.get("n_grid_x", 64),
                    cfl=non_target_params.get("cfl", 0.5),
                    cg_tolerance_values=cg_tolerance_values,
                    start_frame=0,
                    end_frame=end_frame,
                    tolerance_rmse=tolerance_rmse,
                )
                optimal_params = {"cg_tolerance": best_cg_tolerance}

            else:
                raise ValueError(f"Unknown target parameter: {target_param}")

            task_elapsed = time.time() - task_start_time
            total_cost = sum(cost_history)

            # Update statistics
            statistics["total_cost"] += total_cost
            statistics["total_time"] += task_elapsed
            statistics["tasks_completed"] += 1

            if is_converged:
                statistics["convergence_by_precision"][precision]["converged"] += 1
                statistics["convergence_by_profile"][profile]["converged"] += 1

            # Build task result
            task_result = {
                "task_id": f"{profile}_{precision}_{target_param}_{hash(str(non_target_params))}",
                "profile": profile,
                "testcase": testcase,
                "precision_level": precision,
                "tolerance_rmse": tolerance_rmse,
                "target_parameter": target_param,
                "non_target_parameters": non_target_params,
                "search_type": task_config["search_type"],
                "is_converged": is_converged,
                "optimal_parameters": optimal_params,
                "cost_history": cost_history,
                "total_cost": total_cost,
                "execution_time_seconds": task_elapsed,
                "parameter_history": param_history
            }

            if is_converged:
                successful_tasks.append(task_result)
                print(f"✅ Task succeeded: {optimal_params}")
            else:
                failed_tasks.append(task_result)
                print(f"❌ Task failed to converge")

        except Exception as e:
            print(f"❌ Task failed with exception: {e}")
            task_elapsed = time.time() - task_start_time

            failed_tasks.append({
                "task_id": f"{profile}_{precision}_{target_param}",
                "profile": profile,
                "testcase": testcase,
                "precision_level": precision,
                "tolerance_rmse": tolerance_rmse,
                "target_parameter": target_param,
                "is_converged": False,
                "error": str(e),
                "execution_time_seconds": task_elapsed
            })

    # Calculate final statistics
    total_elapsed = time.time() - start_time_total

    print(f"\n{'='*80}")
    print(f"Checkout Complete")
    print(f"{'='*80}")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Successful: {len(successful_tasks)}")
    print(f"Failed: {len(failed_tasks)}")
    print(f"Total cost: {statistics['total_cost']}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"{'='*80}")

    # Print precision-level statistics
    print(f"\nConvergence by Precision Level:")
    for precision, stats in statistics["convergence_by_precision"].items():
        rate = (stats["converged"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {precision}: {stats['converged']}/{stats['total']} ({rate:.1f}%)")

    # Print profile statistics
    print(f"\nConvergence by Profile:")
    for profile, stats in statistics["convergence_by_profile"].items():
        rate = (stats["converged"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  {profile}: {stats['converged']}/{stats['total']} ({rate:.1f}%)")

    # Save datasets
    success_file, failed_file = save_datasets(successful_tasks, failed_tasks, output_dir)

    # Save summary statistics
    summary_file = os.path.join(output_dir, "euler_2d", "summary.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump({
            "solver": "euler_2d",
            "total_tasks": len(all_tasks),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "total_cost": statistics["total_cost"],
            "total_time_seconds": total_elapsed,
            "statistics": {
                "by_precision": dict(statistics["convergence_by_precision"]),
                "by_profile": dict(statistics["convergence_by_profile"])
            }
        }, f, indent=2)

    print(f"\n✅ Summary saved to {summary_file}")

    return successful_tasks, failed_tasks, statistics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Euler 2D solver checkout")
    parser.add_argument(
        "--config",
        type=str,
        default="checkouts/euler_2d.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset",
        help="Output directory for results"
    )
    parser.add_argument(
        "--profiles",
        type=str,
        nargs="+",
        default=None,
        help="Profiles to test (e.g., p1 p2). If not specified, tests all active profiles in config."
    )

    args = parser.parse_args()

    run_checkout(args.config, args.output, args.profiles)
