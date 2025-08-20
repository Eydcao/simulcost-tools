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
from dummy_sols.epoch import find_convergent_npart, find_convergent_nx, find_optimal_dt_multipler, find_optimal_field_order, find_optimal_particle_order
from checkouts.config_utils import load_config, build_target_configs


def save_datasets(successful_tasks, failed_tasks, output_dir):
    """Save successful and failed tasks as separate JSON datasets in subfolders"""
    # Create subfolders for successful and failed tasks
    success_dir = os.path.join(output_dir, "epoch", "successful")
    failed_dir = os.path.join(output_dir, "epoch", "failed")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    
    # Save successful tasks (overwrite existing file)
    success_file = os.path.join(success_dir, "tasks.json")
    with open(success_file, "w") as f:  # "w" mode overwrites existing file
        json.dump({
            "metadata": {
                "solver": "epoch",
                "description": "Successfully converged parameter optimization tasks",
                "total_tasks": len(successful_tasks)
            },
            "tasks": successful_tasks
        }, f, indent=2)
    
    # Save failed tasks (overwrite existing file)
    failed_file = os.path.join(failed_dir, "tasks.json")
    with open(failed_file, "w") as f:  # "w" mode overwrites existing file
        json.dump({
            "metadata": {
                "solver": "epoch", 
                "description": "Failed to converge parameter optimization tasks",
                "total_tasks": len(failed_tasks)
            },
            "tasks": failed_tasks
        }, f, indent=2)
    
    print(f"✅ Saved {len(successful_tasks)} successful tasks to {success_file}")
    print(f"❌ Saved {len(failed_tasks)} failed tasks to {failed_file}")
    
    return success_file, failed_file


def plot_statistics(statistics, output_dir):
    """Plot convergence and optimal parameter statistics"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Convergence rate by precision level and target parameter
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Epoch 1D Dummy Search Statistics", fontsize=16)

    # Plot 1: Convergence rates by precision level
    ax = axes[0, 0]
    precision_levels = list(statistics["convergence_by_precision"].keys())
    convergence_rates = []
    for precision in precision_levels:
        total = statistics["convergence_by_precision"][precision]["total"]
        converged = statistics["convergence_by_precision"][precision]["converged"]
        rate = (converged / total * 100) if total > 0 else 0
        convergence_rates.append(rate)

    ax.bar(precision_levels, convergence_rates, color=["red", "orange", "green"]) # 3 percision levels
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

    ax.bar(target_params, target_rates, color=["blue", "cyan", "purple", "magenta", "orange"]) # 5 parameters
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

    if statistics["optimal_dt_multipler_values"]:

        dt_1st_values= [t[0] for t in list(statistics["optimal_dt_multipler_values"])]

        dt_values, dt_counts = np.unique(dt_1st_values, return_counts=True)
        ax.bar(
            [f"{t:.4f}" for t in dt_values],
            dt_counts,
            alpha=0.7,
            label="dt_multipler parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    if statistics["optimal_nx_values"]:
        nx_values, nx_counts = np.unique(list(statistics["optimal_nx_values"]), return_counts=True)
        ax.bar(
            [str(n) for n in nx_values],
            nx_counts,
            alpha=0.7,
            label="nx parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    if statistics["optimal_npart_values"]:
        npart_values, npart_counts = np.unique(list(statistics["optimal_npart_values"]), return_counts=True)
        ax.bar(
            [str(p) for p in npart_values], npart_counts, alpha=0.7, label="npart parameter", color=colors[color_idx % len(colors)]
        )
        color_idx += 1

    if statistics["optimal_field_order_values"]:
        field_values, field_counts = np.unique(list(statistics["optimal_field_order_values"]), return_counts=True)
        ax.bar(
            [str(f) for f in field_values],
            field_counts,
            alpha=0.7,
            label="field_order parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    if statistics["optimal_particle_order_values"]:
        particle_values, particle_counts = np.unique(list(statistics["optimal_particle_order_values"]), return_counts=True)
        ax.bar(
            [str(p) for p in particle_values],
            particle_counts,
            alpha=0.7,
            label="particle_order parameter",
            color=colors[color_idx % len(colors)],
        )

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
    plt.savefig(os.path.join(output_dir, "epoch_statistics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Create detailed statistics file
    stats_file = os.path.join(output_dir, "epoch_statistics_summary.txt")
    with open(stats_file, "w") as f:
        f.write("=== EPOCH Dummy Search Statistics Summary ===\n\n")

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
        if statistics["optimal_dt_multipler_values"]:
            dt_1st_values= [t[0] for t in list(statistics["optimal_dt_multipler_values"])]

            dt_values, dt_counts = np.unique(dt_1st_values, return_counts=True)
            f.write("   dt_multipler parameter (0-shot):\n")
            for dt, count in zip(dt_values, dt_counts):
                f.write(f"     dt_multipler={dt:.4f}: {count} times\n")

        if statistics["optimal_nx_values"]:
            nx_values, nx_counts = np.unique(list(statistics["optimal_nx_values"]), return_counts=True)
            f.write("   nx parameter (iterative):\n")
            for nx, count in zip(nx_values, nx_counts):
                f.write(f"     nx={nx}: {count} times\n")

        if statistics["optimal_npart_values"]:
            npart_values, npart_counts = np.unique(list(statistics["optimal_npart_values"]), return_counts=True)
            f.write("   npart parameter (iterative):\n")
            for npart, count in zip(npart_values, npart_counts):
                f.write(f"     npart={npart}: {count} times\n")

        if statistics["optimal_field_order_values"]:
            field_values, field_counts = np.unique(list(statistics["optimal_field_order_values"]), return_counts=True)
            f.write("   field_order parameter (0-shot):\n")
            for field, count in zip(field_values, field_counts):
                f.write(f"     field_order={field}: {count} times\n")

        if statistics["optimal_particle_order_values"]:
            particle_values, particle_counts = np.unique(list(statistics["optimal_particle_order_values"]), return_counts=True)
            f.write("   particle_order parameter (0-shot):\n")
            for particle, count in zip(particle_values, particle_counts):
                f.write(f"     particle_order={particle}: {count} times\n")


def main():
    print("=== EPOCH Dummy Solution Generation ===")
    print("Loading configuration from epoch.yaml...")

    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "epoch.yaml")
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
        "optimal_dt_multipler_values": [],
        "optimal_nx_values": [],
        "optimal_npart_values": [],
        "optimal_field_order_values": [],
        "optimal_particle_order_values": [],
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
                    if target_param == "dt_multipler":
                        is_converged, dt_param, cost_history, param_history = find_optimal_dt_multipler(
                            profile=profile,
                            nx=task_params["nx"],
                            npart=task_params["npart"],
                            field_order=task_params["field_order"],
                            particle_order=task_params["particle_order"],
                            search_range_min=target_config["search_range_min"],
                            search_range_max=target_config["search_range_max"],
                            search_range_slice_num=target_config["search_range_slice_num"],
                            tolerance_rmse=precision_vals["tolerance_rmse"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        if dt_param is not None:
                            statistics["optimal_dt_multipler_values"].append(dt_param)
                        optimal_value = dt_param

                    elif target_param == "nx":
                        is_converged, nx_param, cost_history, param_history = find_convergent_nx(
                            profile=profile,
                            dt_multipler=task_params["dt_multipler"],
                            nx=target_config["initial_value"],
                            npart=task_params["npart"],
                            field_order=task_params["field_order"],
                            particle_order=task_params["particle_order"],
                            tolerance_rmse=precision_vals["tolerance_rmse"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        if nx_param is not None:
                            statistics["optimal_nx_values"].append(nx_param)
                        optimal_value = nx_param
                    
                    elif target_param == "npart":
                        is_converged, npart_param, cost_history, param_history = find_convergent_npart(
                            profile=profile,
                            dt_multipler=task_params["dt_multipler"],
                            nx=task_params["nx"],
                            npart=target_config["initial_value"],
                            field_order=task_params["field_order"],
                            particle_order=task_params["particle_order"],
                            tolerance_rmse=precision_vals["tolerance_rmse"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        if npart_param is not None:
                            statistics["optimal_npart_values"].append(npart_param)
                        optimal_value = npart_param

                    elif target_param == "field_order":
                        is_converged, field_params, cost_history, param_history = find_optimal_field_order(
                            profile=profile,
                            dt_multipler=task_params["dt_multipler"],
                            nx=task_params["nx"],
                            npart=task_params["npart"],
                            particle_order=task_params["particle_order"],
                            tolerance_rmse=precision_vals["tolerance_rmse"],
                            possibleOrders=target_config["search_values"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        field_param = field_params[0] if field_params[0] is not None else None
                        if field_param is not None:
                            statistics["optimal_field_order_values"].append(field_param)
                        optimal_value = field_param

                    elif target_param == "particle_order":
                        is_converged, particle_params, cost_history, param_history = find_optimal_particle_order(
                            profile=profile,
                            dt_multipler=task_params["dt_multipler"],
                            nx=task_params["nx"],
                            npart=task_params["npart"],
                            field_order=task_params["field_order"],
                            tolerance_rmse=precision_vals["tolerance_rmse"],
                            possibleOrders=target_config["search_values"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        particle_param = particle_params[0] if particle_params[0] is not None else None
                        if particle_param is not None:
                            statistics["optimal_particle_order_values"].append(particle_param)
                        optimal_value = particle_param

                    # Create task record for dataset
                    task_record = {
                        "solver": "epoch",
                        "target_parameter": target_param,
                        "profile": profile,
                        "precision_config": {
                            "tolerance_rmse": precision_vals["tolerance_rmse"]
                        },
                        "target_config": {
                            "initial_value": target_config.get("initial_value"),
                            "multiplication_factor": target_config.get("multiplication_factor"),
                            "max_iteration_num": target_config.get("max_iteration_num"),
                            "search_range_min": target_config.get("search_range_min"),
                            "search_range_max": target_config.get("search_range_max"),
                            "search_range_slice_num": target_config.get("search_range_slice_num")
                        },
                        "non_target_parameters": task_params.copy(),
                        "results": {
                            "converged": is_converged,
                            "optimal_parameter_value": optimal_value,
                            "total_computational_cost": sum(cost_history) if cost_history else 0,
                            "cost_history": cost_history if cost_history else [],
                            "parameter_history": param_history if param_history else []
                        }
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
            print(f"   {i+1}. {task['profile']} profile, {task['target_parameter']} optimization -> {task['results']['optimal_parameter_value']}")
    
    if len(failed_tasks) > 0:
        print(f"\n📉 Failed Task Examples:")
        for i, task in enumerate(failed_tasks[:3]):  # Show first 3 failed tasks
            print(f"   {i+1}. {task['profile']} profile, {task['target_parameter']} optimization (cost: {task['results']['total_computational_cost']})")

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
