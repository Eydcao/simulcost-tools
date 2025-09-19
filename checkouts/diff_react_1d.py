import itertools
import os
import sys
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dummy_sols.diff_react_1d import find_convergent_cfl, find_convergent_n_space, find_convergent_tolerance, find_convergent_min_step, find_convergent_initial_step_guess
from wrappers.diff_react_1d import run_sim_diff_react_1d
from checkouts.config_utils import load_config, build_target_configs
import yaml


def get_profile_environment_params(profile):
    """Get all environment parameters from profile config file"""
    config_path = f"run_configs/diff_react_1d/{profile}.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Extract environment dependent parameters
            env_params = {
                'reaction_type': config.get('reaction_type'),
                'record_dt': config.get('record_dt'),
                'end_frame': config.get('end_frame'),
                'max_iter': config.get('max_iter')
            }
            
            # Check if all required parameters are present
            missing_params = [k for k, v in env_params.items() if v is None]
            if missing_params:
                raise ValueError(f"Missing parameters in config file {config_path}: {missing_params}")

            if env_params["reaction_type"] == "allee":
                env_params["allee_threshold"] = config.get('allee_threshold')
                
            return env_params
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_path} not found. Please ensure the profile '{profile}' exists.")


def save_datasets(successful_tasks, failed_tasks, output_dir):
    """Save successful and failed tasks as separate JSON datasets in subfolders"""
    # Create subfolders for successful and failed tasks
    success_dir = os.path.join(output_dir, "diff_react_1d", "successful")
    failed_dir = os.path.join(output_dir, "diff_react_1d", "failed")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    # Save successful tasks (overwrite existing file)
    success_file = os.path.join(success_dir, "tasks.json")
    with open(success_file, "w") as f:  # "w" mode overwrites existing file
        json.dump(
            {
                "metadata": {
                    "solver": "diff_react_1d",
                    "description": "Successfully converged parameter optimization tasks",
                    "total_tasks": len(successful_tasks),
                },
                "tasks": successful_tasks,
            },
            f,
            indent=2,
        )

    # Save failed tasks (overwrite existing file)
    failed_file = os.path.join(failed_dir, "tasks.json")
    with open(failed_file, "w") as f:  # "w" mode overwrites existing file
        json.dump(
            {
                "metadata": {
                    "solver": "diff_react_1d",
                    "description": "Failed to converge parameter optimization tasks",
                    "total_tasks": len(failed_tasks),
                },
                "tasks": failed_tasks,
            },
            f,
            indent=2,
        )

    print(f"✅ Saved {len(successful_tasks)} successful tasks to {success_file}")
    print(f"❌ Saved {len(failed_tasks)} failed tasks to {failed_file}")

    return success_file, failed_file


def plot_statistics(statistics, output_dir):
    """Plot convergence and optimal parameter statistics"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Convergence rate by precision level and target parameter
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("DiffReact 1D Dummy Search Statistics", fontsize=16)

    # Plot 1: Convergence rates by precision level
    ax = axes[0, 0]
    precision_levels = list(statistics["convergence_by_precision"].keys())
    convergence_rates = []
    for precision in precision_levels:
        total = statistics["convergence_by_precision"][precision]["total"]
        converged = statistics["convergence_by_precision"][precision]["converged"]
        rate = (converged / total * 100) if total > 0 else 0
        convergence_rates.append(rate)

    ax.bar(precision_levels, convergence_rates, color=["red", "orange", "green"])
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_title("Convergence Rate by Precision Level")
    ax.set_ylim(0, 100)

    # Add text annotations
    for i, rate in enumerate(convergence_rates):
        ax.text(i, rate + 2, f"{rate:.1f}%", ha="center", va="bottom")

    # Plot 2: Convergence rates by target parameter
    ax = axes[0, 1]
    target_params = list(statistics["convergence_by_target"].keys())
    target_rates = []
    for param in target_params:
        total = statistics["convergence_by_target"][param]["total"]
        converged = statistics["convergence_by_target"][param]["converged"]
        rate = (converged / total * 100) if total > 0 else 0
        target_rates.append(rate)

    ax.bar(target_params, target_rates, color=["blue", "cyan", "purple", "pink", "brown"])
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_title("Convergence Rate by Target Parameter")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)

    # Add text annotations
    for i, rate in enumerate(target_rates):
        ax.text(i, rate + 2, f"{rate:.1f}%", ha="center", va="bottom")

    # Plot 3: Optimal parameter frequency (for all tasks)
    ax = axes[1, 0]
    colors = ["skyblue", "lightgreen", "lightcoral", "lightyellow", "lightpink"]
    color_idx = 0

    if statistics["optimal_cfl_values"]:
        cfl_values, cfl_counts = np.unique(list(statistics["optimal_cfl_values"]), return_counts=True)
        ax.bar(
            [f"{c:.4f}" for c in cfl_values],
            cfl_counts,
            alpha=0.7,
            label="CFL parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    if statistics["optimal_n_space_values"]:
        n_space_values, n_space_counts = np.unique(list(statistics["optimal_n_space_values"]), return_counts=True)
        ax.bar(
            [str(n) for n in n_space_values],
            n_space_counts,
            alpha=0.7,
            label="n_space parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    if statistics["optimal_tol_values"]:
        tol_values, tol_counts = np.unique(list(statistics["optimal_tol_values"]), return_counts=True)
        ax.bar(
            [f"{t:.1e}" for t in tol_values],
            tol_counts,
            alpha=0.7,
            label="tolerance parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    if statistics["optimal_min_step_values"]:
        min_step_values, min_step_counts = np.unique(list(statistics["optimal_min_step_values"]), return_counts=True)
        ax.bar(
            [f"{m:.1e}" for m in min_step_values],
            min_step_counts,
            alpha=0.7,
            label="min_step parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    if statistics["optimal_initial_step_guess_values"]:
        initial_step_guess_values, initial_step_guess_counts = np.unique(list(statistics["optimal_initial_step_guess_values"]), return_counts=True)
        ax.bar(
            [f"{i:.4f}" for i in initial_step_guess_values],
            initial_step_guess_counts,
            alpha=0.7,
            label="initial_step_guess parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    ax.set_ylabel("Frequency")
    ax.set_title("Optimal Parameter Values (All Tasks)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    # Plot 4: Cost distribution
    ax = axes[1, 1]
    all_costs = []
    for target in statistics["convergence_by_target"]:
        all_costs.extend(statistics["convergence_by_target"][target]["costs"])

    if all_costs:
        ax.hist(all_costs, bins=20, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Total Cost")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Total Costs")
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diff_react_1d_statistics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Create detailed statistics file
    stats_file = os.path.join(output_dir, "diff_react_1d_statistics_summary.txt")
    with open(stats_file, "w") as f:
        f.write("DIFFREACT 1D DUMMY SOLUTION STATISTICS SUMMARY\n")
        f.write("===============================================\n\n")

        f.write(f"Total tasks executed: {statistics['total_tasks']}\n")
        f.write(f"Successfully converged: {statistics['total_converged']}\n")
        f.write(f"Overall convergence rate: {(statistics['total_converged']/statistics['total_tasks']*100):.2f}%\n\n")

        f.write("CONVERGENCE BY TARGET PARAMETER:\n")
        for param, data in statistics["convergence_by_target"].items():
            rate = (data["converged"] / data["total"] * 100) if data["total"] > 0 else 0
            f.write(f"  {param.upper()}: {rate:.2f}% ({data['converged']}/{data['total']})\n")

        f.write("\nCONVERGENCE BY PRECISION LEVEL:\n")
        for precision, data in statistics["convergence_by_precision"].items():
            rate = (data["converged"] / data["total"] * 100) if data["total"] > 0 else 0
            f.write(f"  {precision.upper()}: {rate:.2f}% ({data['converged']}/{data['total']})\n")

        f.write("\nCONVERGENCE BY PROFILE:\n")
        for profile, data in statistics["convergence_by_profile"].items():
            rate = (data["converged"] / data["total"] * 100) if data["total"] > 0 else 0
            f.write(f"  {profile.upper()}: {rate:.2f}% ({data['converged']}/{data['total']})\n")

        f.write("\nMOST FREQUENT OPTIMAL VALUES:\n")
        if statistics["optimal_cfl_values"]:
            cfl_values, cfl_counts = np.unique(list(statistics["optimal_cfl_values"]), return_counts=True)
            total_cfl = len(statistics["optimal_cfl_values"])
            for cfl, count in zip(cfl_values, cfl_counts):
                percentage = (count / total_cfl * 100) if total_cfl > 0 else 0
                f.write(f"  cfl: {cfl:.4f} (appears {percentage:.1f}% of the time)\n")

        if statistics["optimal_n_space_values"]:
            n_space_values, n_space_counts = np.unique(list(statistics["optimal_n_space_values"]), return_counts=True)
            total_nspace = len(statistics["optimal_n_space_values"])
            for n_space, count in zip(n_space_values, n_space_counts):
                percentage = (count / total_nspace * 100) if total_nspace > 0 else 0
                f.write(f"  n_space: {n_space} (appears {percentage:.1f}% of the time)\n")

        if statistics["optimal_tol_values"]:
            tol_values, tol_counts = np.unique(list(statistics["optimal_tol_values"]), return_counts=True)
            total_tol = len(statistics["optimal_tol_values"])
            for tol, count in zip(tol_values, tol_counts):
                percentage = (count / total_tol * 100) if total_tol > 0 else 0
                f.write(f"  tol: {tol:.1e} (appears {percentage:.1f}% of the time)\n")

        if statistics["optimal_min_step_values"]:
            min_step_values, min_step_counts = np.unique(list(statistics["optimal_min_step_values"]), return_counts=True)
            total_min_step = len(statistics["optimal_min_step_values"])
            for min_step, count in zip(min_step_values, min_step_counts):
                percentage = (count / total_min_step * 100) if total_min_step > 0 else 0
                f.write(f"  min_step: {min_step:.1e} (appears {percentage:.1f}% of the time)\n")

        if statistics["optimal_initial_step_guess_values"]:
            initial_step_guess_values, initial_step_guess_counts = np.unique(list(statistics["optimal_initial_step_guess_values"]), return_counts=True)
            total_initial_step_guess = len(statistics["optimal_initial_step_guess_values"])
            for initial_step_guess, count in zip(initial_step_guess_values, initial_step_guess_counts):
                percentage = (count / total_initial_step_guess * 100) if total_initial_step_guess > 0 else 0
                f.write(f"  initial_step_guess: {initial_step_guess:.4f} (appears {percentage:.1f}% of the time)\n")


def main():
    print("=== DiffReact 1D Dummy Solution Generation ===")
    print("Loading configuration from diff_react_1d.yaml...")

    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "diff_react_1d.yaml")
    config = load_config(config_path)
    print("✅ Configuration loaded successfully")

    # Extract configuration sections
    precision_configs = {}
    for name, info in config["precision_levels"].items():
        # Only process precision levels with numeric values (skip placeholders)
        if isinstance(info["tolerance_rmse"], (int, float)):
            precision_configs[name] = {
                "tolerance_rmse": info["tolerance_rmse"],
            }

    profiles = config["profiles"]["active_profiles"]
    target_configs = build_target_configs(config)

    print(f"📊 Active precision levels: {list(precision_configs.keys())}")
    print(f"📁 Active profiles: {profiles}")
    print(f"🎯 Target parameters: {list(target_configs.keys())}")
    print("Generating all cached results for LLM automation tasks...")

    # Change to repository root directory
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    print(f"Working directory: {os.getcwd()}")

    # Initialize statistics tracking
    statistics = {
        "total_tasks": 0,
        "total_converged": 0,
        "convergence_by_precision": defaultdict(lambda: {"total": 0, "converged": 0}),
        "convergence_by_target": defaultdict(lambda: {"total": 0, "converged": 0, "costs": []}),
        "convergence_by_profile": defaultdict(lambda: {"total": 0, "converged": 0}),
        "optimal_cfl_values": [],
        "optimal_n_space_values": [],
        "optimal_tol_values": [],
        "optimal_min_step_values": [],
        "optimal_initial_step_guess_values": [],
    }

    # Initialize task collection for datasets
    successful_tasks = []
    failed_tasks = []

    # Generate all task combinations (process high precision first)
    precision_order = ["high", "medium", "low"]
    for precision_name in precision_order:
        if precision_name not in precision_configs:
            continue
        precision_vals = precision_configs[precision_name]
        print(f"\n--- Processing {precision_name.upper()} precision ---")

        for profile in profiles:
            print(f"  Profile: {profile}")
            
            # Load environment parameters from profile config
            env_params = get_profile_environment_params(profile)

            for target_param, target_config in target_configs.items():
                print(f"    Target parameter: {target_param}")

                # Get all non-target parameter combinations
                non_target_params = target_config["non_target_parameters"]
                
                # Handle both list format (matched combinations) and dict format (old format)
                if isinstance(non_target_params, list):
                    # List of dictionaries (matched combinations)
                    param_combinations = non_target_params
                else:
                    # Dictionary format (old format) - generate all combinations
                    param_names = list(non_target_params.keys())
                    param_values = [
                        non_target_params[name] if isinstance(non_target_params[name], list) else [non_target_params[name]]
                        for name in param_names
                    ]
                    param_combinations = [dict(zip(param_names, combination)) for combination in itertools.product(*param_values)]

                # Iterate through each parameter combination
                for task_params in param_combinations:
                    # Add environment parameters to task_params
                    task_params.update(env_params)

                    print(f"      Running {target_param} search with params: {task_params}")
                    # Call appropriate search function based on target parameter
                    if target_param == "cfl":
                        is_converged, best_param, cost_history, param_history = find_convergent_cfl(
                            profile=profile,
                            initial_cfl=target_config["initial_value"],
                            initial_n_space=task_params["n_space"],
                            tolerance=precision_vals["tolerance_rmse"],
                            max_iter=target_config["max_iteration_num"],
                            multiplication_factor=target_config["multiplication_factor"],
                        )
                        if best_param is not None:
                            statistics["optimal_cfl_values"].append(best_param)

                    elif target_param == "n_space":
                        is_converged, best_param, cost_history, param_history = find_convergent_n_space(
                            profile=profile,
                            initial_n_space=target_config["initial_value"],
                            cfl=task_params["cfl"],
                            tolerance=precision_vals["tolerance_rmse"],
                            max_iter=target_config["max_iteration_num"],
                            multiplication_factor=target_config["multiplication_factor"],
                        )
                        if best_param is not None:
                            statistics["optimal_n_space_values"].append(best_param)

                    elif target_param == "tol":
                        is_converged, best_param, cost_history, param_history = find_convergent_tolerance(
                            profile=profile,
                            initial_tol=target_config["initial_value"],
                            n_space=task_params["n_space"],
                            cfl=task_params["cfl"],
                            tolerance=precision_vals["tolerance_rmse"],
                            max_iter=target_config["max_iteration_num"],
                            multiplication_factor=target_config["multiplication_factor"],
                        )
                        if best_param is not None:
                            statistics["optimal_tol_values"].append(best_param)

                    elif target_param == "min_step":
                        is_converged, best_param, cost_history, param_history = find_convergent_min_step(
                            profile=profile,
                            initial_min_step=target_config["initial_value"],
                            n_space=task_params["n_space"],
                            cfl=task_params["cfl"],
                            tolerance=precision_vals["tolerance_rmse"],
                            max_iter=target_config["max_iteration_num"],
                            multiplication_factor=target_config["multiplication_factor"],
                        )
                        if best_param is not None:
                            statistics["optimal_min_step_values"].append(best_param)

                    elif target_param == "initial_step_guess":
                        # Handle 0-shot search with exact_values
                        if "exact_values" in target_config:
                            initial_step_guess_values = target_config["exact_values"]
                            print(f"Using exact_values from YAML for initial_step_guess: {initial_step_guess_values}")
                            
                            # Test each exact value and find the best one
                            best_converged = False
                            best_param = None
                            best_cost = float('inf')
                            cost_history = []
                            param_history = []
                            
                            for initial_step_guess in initial_step_guess_values:
                                try:
                                    cost = run_sim_diff_react_1d(
                                        profile=profile,
                                        n_space=task_params["n_space"],
                                        cfl=task_params["cfl"],
                                        tol=task_params["tol"],
                                        min_step=task_params["min_step"],
                                        initial_step_guess=initial_step_guess,
                                        reaction_type=task_params.get("reaction_type", "fisher"),
                                        allee_threshold=task_params.get("allee_threshold")
                                    )
                                    cost_history.append(cost)
                                    param_history.append(initial_step_guess)
                                    
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_param = initial_step_guess
                                        best_converged = True
                                        
                                except Exception as e:
                                    print(f"Simulation failed with initial_step_guess {initial_step_guess}: {e}")
                                    cost_history.append(float('inf'))
                                    param_history.append(initial_step_guess)
                            
                            is_converged = best_converged
                        else:
                            # Fallback to iterative search
                            is_converged, best_param, cost_history, param_history = find_convergent_initial_step_guess(
                                profile=profile,
                                initial_step_guess=target_config["initial_value"],
                                n_space=task_params["n_space"],
                                cfl=task_params["cfl"],
                                tolerance=precision_vals["tolerance_rmse"],
                                max_iter=target_config["max_iteration_num"],
                                multiplication_factor=target_config["multiplication_factor"],
                            )
                        
                        if best_param is not None:
                            statistics["optimal_initial_step_guess_values"].append(best_param)

                    # Create task record for dataset
                    task_record = {
                        "solver": "diff_react_1d",
                        "target_parameter": target_param,
                        "profile": profile,
                        "precision_level": precision_name,
                        "precision_config": {"tolerance_rmse": precision_vals["tolerance_rmse"]},
                        "target_config": {
                            "initial_value": target_config.get("initial_value"),
                            "multiplication_factor": target_config.get("multiplication_factor"),
                            "max_iteration_num": target_config.get("max_iteration_num"),
                        },
                        "non_target_parameters": task_params.copy(),
                        "results": {
                            "converged": is_converged,
                            "optimal_parameter_value": best_param,
                            "total_computational_cost": sum(cost_history) if cost_history else 0,
                            "cost_history": cost_history if cost_history else [],
                            "parameter_history": param_history if param_history else [],
                        },
                    }

                    # Add task to appropriate dataset
                    if is_converged:
                        successful_tasks.append(task_record)
                    else:
                        failed_tasks.append(task_record)

                    # Update statistics
                    total_cost = sum(cost_history) if cost_history else 0
                    statistics["total_tasks"] += 1
                    statistics["convergence_by_precision"][precision_name]["total"] += 1
                    statistics["convergence_by_target"][target_param]["total"] += 1
                    statistics["convergence_by_profile"][profile]["total"] += 1
                    statistics["convergence_by_target"][target_param]["costs"].append(total_cost)

                    if is_converged:
                        statistics["total_converged"] += 1
                        statistics["convergence_by_precision"][precision_name]["converged"] += 1
                        statistics["convergence_by_target"][target_param]["converged"] += 1
                        statistics["convergence_by_profile"][profile]["converged"] += 1
                        print(f"      ✅ SUCCESS: Found {target_param}={best_param}, cost={total_cost}")
                    else:
                        print(f"      ❌ FAILED: No convergence, cost={total_cost}")

    print(f"\n=== Generation Complete ===")
    print(f"Total tasks: {statistics['total_tasks']}")
    print(f"Successfully converged: {statistics['total_converged']}")
    print(f"Overall convergence rate: {(statistics['total_converged']/statistics['total_tasks']*100):.2f}%")

    # Generate statistics plots
    output_dir = os.path.join(repo_root, "outputs", "statistics")
    plot_statistics(statistics, output_dir)
    print(f"📊 Statistics plots saved to: {output_dir}")

    # Save datasets
    dataset_dir = os.path.join(repo_root, "dataset")
    success_file, failed_file = save_datasets(successful_tasks, failed_tasks, dataset_dir)
    print(f"💾 Dataset files saved to: {dataset_dir}")
    print(f"   ✅ Successful tasks: {len(successful_tasks)} tasks")
    print(f"   ❌ Failed tasks: {len(failed_tasks)} tasks")

    # Display dataset summary
    if len(successful_tasks) > 0:
        print(f"\n📈 Successful Task Examples:")
        for i, task in enumerate(successful_tasks[:3]):  # Show first 3 successful tasks
            print(
                f"   {i+1}. {task['profile']} profile, {task['target_parameter']} optimization -> {task['results']['optimal_parameter_value']}"
            )

    if len(failed_tasks) > 0:
        print(f"\n📉 Failed Task Examples:")
        for i, task in enumerate(failed_tasks[:3]):  # Show first 3 failed tasks
            print(
                f"   {i+1}. {task['profile']} profile, {task['target_parameter']} optimization (cost: {task['results']['total_computational_cost']})"
            )

    # Expected task calculation for verification
    expected_total = 0
    for target_param, target_config in target_configs.items():
        param_values = []
        for name in target_config["non_target_parameters"]:
            value = target_config["non_target_parameters"][name]
            if isinstance(value, list):
                param_values.append(value)
            else:
                param_values.append([value])

        combinations_per_target = 1
        for values in param_values:
            combinations_per_target *= len(values)
        expected_total += len(profiles) * combinations_per_target

    print(f"\nTask breakdown:")
    for target_param, target_config in target_configs.items():
        param_values = []
        for name in target_config["non_target_parameters"]:
            value = target_config["non_target_parameters"][name]
            if isinstance(value, list):
                param_values.append(value)
            else:
                param_values.append([value])

        combinations_per_target = 1
        for values in param_values:
            combinations_per_target *= len(values)
        tasks_for_param = len(profiles) * combinations_per_target
        print(f"  {target_param}: {len(profiles)} profiles × {combinations_per_target} combos = {tasks_for_param}")
    print(f"  Expected total per precision: {expected_total}")
    print(f"  Expected total across {len(precision_configs)} precisions: {expected_total * len(precision_configs)}")
    print(f"  Actual total: {statistics['total_tasks']}")


if __name__ == "__main__":
    main()
