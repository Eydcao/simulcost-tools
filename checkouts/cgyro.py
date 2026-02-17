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
from dummy_sols.cgyro import (
    find_convergent_nradial,
    find_convergent_ntheta,
    find_convergent_error_tol,
    find_convergent_freq_tol,
    find_convergent_delta_t
)
from checkouts.config_utils import load_config, build_target_configs


def save_datasets(successful_tasks, failed_tasks, output_dir):
    """Save successful and failed tasks as separate JSON datasets in subfolders"""
    # Create subfolders for successful and failed tasks
    success_dir = os.path.join(output_dir, "cgyro", "successful")
    failed_dir = os.path.join(output_dir, "cgyro", "failed")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    # Save successful tasks (overwrite existing file)
    success_file = os.path.join(success_dir, "tasks.json")
    with open(success_file, "w") as f:  # "w" mode overwrites existing file
        json.dump(
            {
                "metadata": {
                    "solver": "cgyro",
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
                    "solver": "cgyro",
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
    fig.suptitle("CGYRO Dummy Search Statistics", fontsize=16)

    # Plot 1: Convergence rates by precision level
    ax = axes[0, 0]
    precision_levels = list(statistics["convergence_by_precision"].keys())
    convergence_rates = []
    for precision in precision_levels:
        total = statistics["convergence_by_precision"][precision]["total"]
        converged = statistics["convergence_by_precision"][precision]["converged"]
        rate = (converged / total * 100) if total > 0 else 0
        convergence_rates.append(rate)

    ax.bar(precision_levels, convergence_rates, color=["red", "orange", "green"])  # 3 percision levels
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

    ax.bar(target_params, target_rates, color=["blue", "cyan", "purple"])  # 3 parameters
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_title("Convergence Rate by Target Parameter")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)

    # Add text annotations
    for i, rate in enumerate(target_rates):
        ax.text(i, rate + 2, f"{rate:.1f}%", ha="center", va="bottom")

    # Plot 3: Optimal parameter frequency (for all tasks)
    ax = axes[1, 0]
    colors = ["skyblue", "lightgreen", "lightcoral", "gold", "pink"]
    color_idx = 0

    if statistics["optimal_nradial_values"]:
        nradial_values, nradial_counts = np.unique(list(statistics["optimal_nradial_values"]), return_counts=True)
        ax.bar(
            [str(n) for n in nradial_values],
            nradial_counts,
            alpha=0.7,
            label="n_radial parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    if statistics["optimal_ntheta_values"]:
        ntheta_values, ntheta_counts = np.unique(list(statistics["optimal_ntheta_values"]), return_counts=True)
        ax.bar(
            [str(n) for n in ntheta_values],
            ntheta_counts,
            alpha=0.7,
            label="n_theta parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    if statistics["optimal_error_tol_values"]:
        error_tol_values, error_tol_counts = np.unique(list(statistics["optimal_error_tol_values"]), return_counts=True)
        ax.bar(
            [str(p) for p in error_tol_values],
            error_tol_counts,
            alpha=0.7,
            label="error_tol parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1
    
    if statistics["optimal_freq_tol_values"]:
        freq_tol_values, freq_tol_counts = np.unique(list(statistics["optimal_freq_tol_values"]), return_counts=True)
        ax.bar(
            [str(p) for p in freq_tol_values],
            freq_tol_counts,
            alpha=0.7,
            label="freq_tol parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1
    
    if statistics["optimal_delta_t_values"]:
        delta_t_values, delta_t_counts = np.unique(list(statistics["optimal_delta_t_values"]), return_counts=True)
        ax.bar(
            [str(p) for p in delta_t_values],
            delta_t_counts,
            alpha=0.7,
            label="delta_t parameter",
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
    plt.savefig(os.path.join(output_dir, "cgyro_statistics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Create detailed statistics file
    stats_file = os.path.join(output_dir, "cgyro_statistics_summary.txt")
    with open(stats_file, "w") as f:
        f.write("=== CGYRO Dummy Search Statistics Summary ===\n\n")

        f.write("1. Overall Statistics:\n")
        f.write(f"   Total tasks: {statistics['total_tasks']}\n")
        f.write(f"   Successfully converged: {statistics['total_converged']}\n")
        f.write(
            f"   Overall convergence rate: {(statistics['total_converged']/statistics['total_tasks']*100):.2f}%\n\n"
        )

        f.write("2. Convergence by Precision Level:\n")
        for precision, data in statistics["convergence_by_precision"].items():
            rate = (data["converged"] / data["total"] * 100) if data["total"] > 0 else 0
            f.write(f"   {precision}: {data['converged']}/{data['total']} ({rate:.2f}%)\n")
        f.write("\n")

        f.write("3. Convergence by Target Parameter:\n")
        for param, data in statistics["convergence_by_target"].items():
            rate = (data["converged"] / data["total"] * 100) if data["total"] > 0 else 0
            avg_cost = np.mean(data["costs"]) if data["costs"] else 0
            f.write(f"   {param}: {data['converged']}/{data['total']} ({rate:.2f}%), avg cost: {avg_cost:.0f}\n")
        f.write("\n")

        f.write("4. Convergence by Profile:\n")
        for profile, data in statistics["convergence_by_profile"].items():
            rate = (data["converged"] / data["total"] * 100) if data["total"] > 0 else 0
            f.write(f"   {profile}: {data['converged']}/{data['total']} ({rate:.2f}%)\n")
        f.write("\n")

        f.write("5. Optimal Parameter Frequencies (All Tasks):\n")
        if statistics["optimal_nradial_values"]:
            nradial_values, nradial_counts = np.unique(list(statistics["optimal_nradial_values"]), return_counts=True)
            f.write("   n_radial parameter (iterative):\n")
            for n_radial, count in zip(nradial_values, nradial_counts):
                f.write(f"     n_radial={n_radial}: {count} times\n")

        if statistics["optimal_ntheta_values"]:
            ntheta_values, ntheta_counts = np.unique(list(statistics["optimal_ntheta_values"]), return_counts=True)
            f.write("   n_theta parameter (iterative):\n")
            for n_theta, count in zip(ntheta_values, ntheta_counts):
                f.write(f"     n_theta={n_theta}: {count} times\n")

        if statistics["optimal_error_tol_values"]:
            error_tol_values, error_tol_counts = np.unique(list(statistics["optimal_error_tol_values"]), return_counts=True)
            f.write("   error_tol parameter (0-shot):\n")
            for error_tol, count in zip(error_tol_values, error_tol_counts):
                f.write(f"     error_tol={error_tol}: {count} times\n")
        
        if statistics["optimal_freq_tol_values"]:
            freq_tol_values, freq_tol_counts = np.unique(list(statistics["optimal_freq_tol_values"]), return_counts=True)
            f.write("   freq_tol parameter (0-shot):\n")
            for freq_tol, count in zip(freq_tol_values, freq_tol_counts):
                f.write(f"     freq_tol={freq_tol}: {count} times\n")
        
        if statistics["optimal_delta_t_values"]:
            delta_t_values, delta_t_counts = np.unique(list(statistics["optimal_delta_t_values"]), return_counts=True)
            f.write("   delta_t parameter (0-shot):\n")
            for delta_t, count in zip(delta_t_values, delta_t_counts):
                f.write(f"     delta_t={delta_t}: {count} times\n")


def main():
    print("=== CGYRO Dummy Solution Generation ===")
    print("Loading configuration from cgyro.yaml...")

    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "cgyro.yaml")
    config = load_config(config_path)
    print("✅ Configuration loaded successfully")

    # Extract configuration sections
    precision_configs = {}
    for name, info in config["precision_levels"].items():
        # Only process precision levels with numeric values (skip placeholders)
        if isinstance(info["comparison_tolerance"], (int, float)):
            precision_configs[name] = {
                "comparison_tolerance": info["comparison_tolerance"],
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
        "optimal_nradial_values": [],
        "optimal_ntheta_values": [],
        "optimal_error_tol_values": [],
        "optimal_freq_tol_values": [],
        "optimal_delta_t_values": []
    }

    # Initialize task collection for datasets
    successful_tasks = []
    failed_tasks = []

    # Generate all task combinations
    for precision_name, precision_vals in precision_configs.items():
        print(f"\n--- Processing {precision_name.upper()} precision ---")

        for profile in profiles:
            print(f"  Profile: {profile}")

            for target_param, target_config in target_configs.items():
                print(f"    Target parameter: {target_param}")

                # Get all non-target parameter names and their value lists
                non_target_params = target_config["non_target_parameters"]
                param_names = list(non_target_params.keys())
                param_values = [non_target_params[name] for name in param_names]

                # Generate all combinations using nested loops (cartesian product)
                for combination in itertools.product(*param_values):
                    # Build parameters dictionary
                    task_params = dict(zip(param_names, combination))

                    print(f"      Running {target_param} search with params: {task_params}")
                    # Call appropriate search function based on target parameter
                    if target_param == "n_radial":
                        is_converged, nradial_param, cost_history, param_history = find_convergent_nradial(
                            profile=profile,
                            n_radial=target_config["initial_value"],
                            n_theta=task_params["n_theta"],
                            error_tol=task_params["error_tol"],
                            freq_tol=task_params["freq_tol"],
                            delta_t=task_params["delta_t"],
                            comparison_tolerance=precision_vals["comparison_tolerance"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        if nradial_param is not None:
                            statistics["optimal_nradial_values"].append(nradial_param)
                        optimal_value = nradial_param
                    
                    elif target_param == "n_theta":
                        is_converged, ntheta_param, cost_history, param_history = find_convergent_ntheta(
                            profile=profile,
                            n_radial=task_params["n_radial"],
                            n_theta=target_config["initial_value"],
                            error_tol=task_params["error_tol"],
                            freq_tol=task_params["freq_tol"],
                            delta_t=task_params["delta_t"],
                            comparison_tolerance=precision_vals["comparison_tolerance"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        if ntheta_param is not None:
                            statistics["optimal_ntheta_values"].append(ntheta_param)
                        optimal_value = ntheta_param
                    
                    elif target_param == "error_tol":
                        is_converged, error_tol_param, cost_history, param_history = find_convergent_error_tol(
                            profile=profile,
                            n_radial=task_params["n_radial"],
                            n_theta=task_params["n_theta"],
                            error_tol=target_config["initial_value"],
                            freq_tol=task_params["freq_tol"],
                            delta_t=task_params["delta_t"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        if error_tol_param is not None:
                            statistics["optimal_error_tol_values"].append(error_tol_param)
                        optimal_value = error_tol_param
                    
                    elif target_param == "freq_tol":
                        is_converged, freq_tol_param, cost_history, param_history = find_convergent_freq_tol(
                            profile=profile,
                            n_radial=task_params["n_radial"],
                            n_theta=task_params["n_theta"],
                            error_tol=task_params["error_tol"],
                            freq_tol=target_config["initial_value"],
                            delta_t=task_params["delta_t"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        if freq_tol_param is not None:
                            statistics["optimal_freq_tol_values"].append(freq_tol_param)
                        optimal_value = freq_tol_param
                    
                    elif target_param == "delta_t":
                        is_converged, delta_t_param, cost_history, param_history = find_convergent_delta_t(
                            profile=profile,
                            n_radial=task_params["n_radial"],
                            n_theta=task_params["n_theta"],
                            error_tol=task_params["error_tol"],
                            freq_tol=task_params["freq_tol"],
                            delta_t=target_config["initial_value"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        if delta_t_param is not None:
                            statistics["optimal_delta_t_values"].append(delta_t_param)
                        optimal_value = delta_t_param
                    
                    # Create task record for dataset
                    task_record = {
                        "solver": "cgyro",
                        "target_parameter": target_param,
                        "profile": profile,
                        "precision_config": {"comparison_tolerance": precision_vals["comparison_tolerance"]},
                        "target_config": {
                            "initial_value": target_config.get("initial_value"),
                            "multiplication_factor": target_config.get("multiplication_factor"),
                            "max_iteration_num": target_config.get("max_iteration_num"),
                            "search_range_min": target_config.get("search_range_min"),
                            "search_range_max": target_config.get("search_range_max"),
                            "search_range_slice_num": target_config.get("search_range_slice_num"),
                        },
                        "non_target_parameters": task_params.copy(),
                        "results": {
                            "converged": is_converged,
                            "optimal_parameter_value": optimal_value,
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
                        print(f"      ✅ SUCCESS: Found {target_param}={optimal_value}, cost={total_cost}")
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
        param_values = [target_config["non_target_parameters"][name] for name in target_config["non_target_parameters"]]
        combinations_per_target = 1
        for values in param_values:
            combinations_per_target *= len(values)
        expected_total += len(profiles) * combinations_per_target

    print(f"\nTask breakdown:")
    for target_param, target_config in target_configs.items():
        param_values = [target_config["non_target_parameters"][name] for name in target_config["non_target_parameters"]]
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
