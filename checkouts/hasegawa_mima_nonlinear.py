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
from dummy_sols.hasegawa_mima_nonlinear import find_convergent_N, find_convergent_dt
from checkouts.config_utils import load_config, build_target_configs


def save_datasets(successful_tasks, failed_tasks, output_dir):
    """Save successful and failed tasks as separate JSON datasets in subfolders"""
    # Create subfolders for successful and failed tasks
    success_dir = os.path.join(output_dir, "hasegawa_mima_nonlinear", "successful")
    failed_dir = os.path.join(output_dir, "hasegawa_mima_nonlinear", "failed")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    # Save successful tasks (overwrite existing file)
    success_file = os.path.join(success_dir, "tasks.json")
    with open(success_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "solver": "hasegawa_mima_nonlinear",
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
    with open(failed_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "solver": "hasegawa_mima_nonlinear",
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
    fig.suptitle("Hasegawa-Mima Nonlinear Dummy Search Statistics", fontsize=16)

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

    ax.bar(target_params, target_rates, color=["blue", "cyan"])
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_title("Convergence Rate by Target Parameter")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)

    # Add text annotations
    for i, rate in enumerate(target_rates):
        ax.text(i, rate + 2, f"{rate:.1f}%", ha="center", va="bottom")

    # Plot 3: Optimal parameter values
    ax = axes[1, 0]
    optimal_values = defaultdict(list)
    for task in statistics["successful_tasks"]:
        param_name = task["target_parameter"]
        optimal_val = task["results"]["optimal_parameter_value"]
        if optimal_val is not None:
            optimal_values[param_name].append(optimal_val)

    # Create histogram for optimal values
    param_names = list(optimal_values.keys())
    colors = ["lightblue", "lightcoral"]
    for i, param in enumerate(param_names):
        values = optimal_values[param]
        if param == "N":
            # For N, show as discrete bins
            unique_vals, counts = np.unique(values, return_counts=True)
            ax.bar(
                [f"{param}\n{val}" for val in unique_vals], counts, color=colors[i % len(colors)], alpha=0.7, width=0.8
            )
        else:
            # For dt, show as histogram
            ax.hist(values, bins=5, alpha=0.7, label=f"{param} parameter", color=colors[i % len(colors)])

    ax.set_ylabel("Frequency")
    ax.set_title("Optimal Parameter Values (All Tasks)")
    ax.tick_params(axis="x", rotation=45)

    # Plot 4: Distribution of total costs
    ax = axes[1, 1]
    costs = [
        task["results"]["total_computational_cost"]
        for task in statistics["successful_tasks"]
        if task["results"]["total_computational_cost"]
    ]

    if costs:
        ax.hist(costs, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax.set_xlabel("Total Cost")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Total Costs")
        ax.set_yscale("log")

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "hasegawa_mima_nonlinear_statistics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Statistics plot saved to {plot_path}")
    return plot_path


def generate_statistics_summary(statistics, output_dir):
    """Generate text summary of statistics"""
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "hasegawa_mima_nonlinear_statistics_summary.txt")

    with open(summary_path, "w") as f:
        f.write("=== Hasegawa-Mima Nonlinear Dummy Search Statistics Summary ===\n\n")

        # Overall statistics
        total_tasks = statistics["total_tasks"]
        successful_tasks = len(statistics["successful_tasks"])
        convergence_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0

        f.write("1. Overall Statistics:\n")
        f.write(f"   Total tasks: {total_tasks}\n")
        f.write(f"   Successfully converged: {successful_tasks}\n")
        f.write(f"   Overall convergence rate: {convergence_rate:.2f}%\n\n")

        # Convergence by precision
        f.write("2. Convergence by Precision Level:\n")
        for precision, data in statistics["convergence_by_precision"].items():
            rate = (data["converged"] / data["total"] * 100) if data["total"] > 0 else 0
            f.write(f"   {precision}: {data['converged']}/{data['total']} ({rate:.2f}%)\n")
        f.write("\n")

        # Convergence by target parameter
        f.write("3. Convergence by Target Parameter:\n")
        for param, data in statistics["convergence_by_target"].items():
            rate = (data["converged"] / data["total"] * 100) if data["total"] > 0 else 0
            avg_cost = data["avg_cost"]
            f.write(f"   {param}: {data['converged']}/{data['total']} ({rate:.2f}%), avg cost: {avg_cost}\n")
        f.write("\n")

        # Optimal parameter frequencies
        f.write("4. Optimal Parameter Frequencies (All Tasks):\n")
        optimal_counts = defaultdict(lambda: defaultdict(int))
        for task in statistics["successful_tasks"]:
            param_name = task["target_parameter"]
            optimal_val = task["results"]["optimal_parameter_value"]
            if optimal_val is not None:
                optimal_counts[param_name][optimal_val] += 1

        for param, counts in optimal_counts.items():
            f.write(f"   {param} parameter:\n")
            for value, count in sorted(counts.items()):
                if param == "dt":
                    f.write(f"     {param}={value:.2e}: {count} times\n")
                else:
                    f.write(f"     {param}={value}: {count} times\n")
        f.write("\n")

    print(f"📝 Statistics summary saved to {summary_path}")
    return summary_path


def main():
    print("🚀 Starting Hasegawa-Mima Nonlinear Parameter Search Checkout")
    print("Loading configuration from hasegawa_mima_nonlinear.yaml...")

    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "hasegawa_mima_nonlinear.yaml")
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

    # Change to repository root directory
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    print(f"Working directory: {os.getcwd()}")

    successful_tasks = []
    failed_tasks = []
    task_id = 0

    total_tasks = (
        len(profiles)
        * len(precision_configs)
        * sum(
            len(list(itertools.product(*config["non_target_parameters"].values())))
            for config in target_configs.values()
        )
    )

    print(f"📋 Total tasks to process: {total_tasks}")

    start_time = time.time()

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
                    task_id += 1

                    # Build parameters dictionary
                    task_params = dict(zip(param_names, combination))
                    initial_params = {target_param: target_config["initial_value"]}
                    initial_params.update(task_params)

                    print(
                        f"      Task {task_id}/{total_tasks}: Running {target_param} search with params: {task_params}"
                    )

                    try:
                        # Call appropriate search function based on target parameter
                        if target_param == "N":
                            is_converged, best_param, cost_history, param_history = find_convergent_N(
                                profile=profile,
                                N=target_config["initial_value"],
                                dt=task_params["dt"],
                                tolerance_rmse=precision_vals["tolerance_rmse"],
                                multiplication_factor=target_config["multiplication_factor"],
                                max_iteration_num=target_config["max_iteration_num"],
                            )
                        elif target_param == "dt":
                            is_converged, best_param, cost_history, param_history = find_convergent_dt(
                                profile=profile,
                                N=task_params["N"],
                                dt=target_config["initial_value"],
                                tolerance_rmse=precision_vals["tolerance_rmse"],
                                multiplication_factor=target_config["multiplication_factor"],
                                max_iteration_num=target_config["max_iteration_num"],
                            )
                        else:
                            raise ValueError(f"Unsupported target parameter: {target_param}")

                        # Create task record
                        task = {
                            "task_id": task_id,
                            "solver": "hasegawa_mima_nonlinear",
                            "profile": profile,
                            "precision_level": precision_name,
                            "tolerance_rmse": precision_vals["tolerance_rmse"],
                            "target_parameter": target_param,
                            "target_config": {
                                "initial_value": target_config["initial_value"],
                                "multiplication_factor": target_config["multiplication_factor"],
                                "max_iteration_num": target_config["max_iteration_num"],
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

                        if is_converged:
                            successful_tasks.append(task)
                            print(f"      ✅ SUCCESS: {target_param}={best_param}")
                        else:
                            failed_tasks.append(task)
                            print(f"      ❌ FAILED: No convergence")

                    except Exception as e:
                        print(f"      💥 Error: {str(e)}")
                        task = {
                            "task_id": task_id,
                            "solver": "hasegawa_mima_nonlinear",
                            "profile": profile,
                            "precision_level": precision_name,
                            "tolerance_rmse": precision_vals["tolerance_rmse"],
                            "target_parameter": target_param,
                            "non_target_parameters": task_params.copy(),
                            "error": str(e),
                        }
                        failed_tasks.append(task)

    elapsed_time = time.time() - start_time
    print(f"\n🏁 Checkout completed in {elapsed_time:.2f} seconds")
    print(f"✅ Successful tasks: {len(successful_tasks)}")
    print(f"❌ Failed tasks: {len(failed_tasks)}")
    print(f"Overall convergence rate: {(len(successful_tasks)/total_tasks*100):.2f}%")

    # Generate statistics
    statistics = {
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "convergence_by_precision": {},
        "convergence_by_target": {},
    }

    # Calculate convergence by precision level
    for precision in precision_configs.keys():
        precision_tasks = [t for t in successful_tasks + failed_tasks if t["precision_level"] == precision]
        precision_successful = [t for t in successful_tasks if t["precision_level"] == precision]
        statistics["convergence_by_precision"][precision] = {
            "total": len(precision_tasks),
            "converged": len(precision_successful),
        }

    # Calculate convergence by target parameter
    for param in target_configs.keys():
        param_tasks = [t for t in successful_tasks + failed_tasks if t["target_parameter"] == param]
        param_successful = [t for t in successful_tasks if t["target_parameter"] == param]
        avg_cost = 0
        if param_successful:
            costs = [
                t["results"]["total_computational_cost"]
                for t in param_successful
                if t["results"]["total_computational_cost"]
            ]
            avg_cost = np.mean(costs) if costs else 0

        statistics["convergence_by_target"][param] = {
            "total": len(param_tasks),
            "converged": len(param_successful),
            "avg_cost": int(avg_cost),
        }

    # Save datasets and generate outputs
    output_dir = os.path.join(repo_root, "dataset")
    save_datasets(successful_tasks, failed_tasks, output_dir)

    # Generate statistics outputs
    stats_dir = os.path.join(repo_root, "outputs", "statistics")
    plot_statistics(statistics, stats_dir)
    generate_statistics_summary(statistics, stats_dir)

    print(f"\n🎯 Checkout generation complete!")
    print(f"📊 Statistics plots saved to: {stats_dir}")
    print(f"💾 Dataset files saved to: {output_dir}")

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
    print(f"  Actual total: {total_tasks}")


if __name__ == "__main__":
    main()
