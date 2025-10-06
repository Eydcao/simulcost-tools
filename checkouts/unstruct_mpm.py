#!/usr/bin/env python3
"""
Unstruct MPM Dummy Solution Generation Script

This script generates dummy solutions for all parameter combinations defined in the
unstruct_mpm.yaml file. It uses direct Python function calls instead of subprocess
for better performance and to capture return statistics.
"""

import itertools
import os
import sys
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkouts.config_utils import load_config, build_target_configs
from dummy_sols.unstruct_mpm import (
    find_convergent_nx, find_convergent_n_part, find_convergent_cfl
)
import yaml


def get_case_from_profile(profile):
    """Get the case from the profile configuration file"""
    config_path = f"run_configs/unstruct_mpm/{profile}.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'case' in config:
                return config['case']
            else:
                raise ValueError(f"case not found in config file {config_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_path} not found. Please ensure the profile '{profile}' exists.")


def save_datasets(successful_tasks, failed_tasks, output_dir):
    """Save successful and failed tasks as separate JSON datasets in subfolders"""
    # Create subfolders for successful and failed tasks
    success_dir = os.path.join(output_dir, "unstruct_mpm", "successful")
    failed_dir = os.path.join(output_dir, "unstruct_mpm", "failed")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    
    # Save successful tasks (overwrite existing file)
    success_file = os.path.join(success_dir, "tasks.json")
    with open(success_file, "w") as f:  # "w" mode overwrites existing file
        json.dump({
            "metadata": {
                "solver": "unstruct_mpm",
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
                "solver": "unstruct_mpm", 
                "description": "Failed to converge parameter optimization tasks",
                "total_tasks": len(failed_tasks)
            },
            "tasks": failed_tasks
        }, f, indent=2)
    
    print(f"✅ Saved {len(successful_tasks)} successful tasks to {success_file}")
    print(f"❌ Saved {len(failed_tasks)} failed tasks to {failed_file}")
    
    return success_file, failed_file


def calculate_quality(nx, profile):
    """Calculate quality based on nx and profile.
    
    Args:
        nx: Grid resolution parameter
        profile: Profile name (p1, p2, p3)
    
    Returns:
        float: Quality value
    """
    if profile == "p1":
        return 0.5 * nx / 11
    elif profile == "p2":
        return nx / 35
    elif profile == "p3":
        return 0.025 * nx
    else:
        raise ValueError(f"Unknown profile: {profile}")


def plot_statistics(statistics, output_dir):
    """Plot convergence and optimal parameter statistics"""
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Unstruct MPM Dummy Search Statistics", fontsize=16)

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

    ax.bar(target_params, target_rates, color=["blue", "cyan", "purple", "magenta"])
    ax.set_ylabel("Convergence Rate (%)")
    ax.set_title("Convergence Rate by Target Parameter")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)

    # Add text annotations
    for i, rate in enumerate(target_rates):
        ax.text(i, rate + 2, f"{rate:.1f}%", ha="center", va="bottom")

    # Plot 3: Optimal parameter frequency (for all tasks)
    ax = axes[1, 0]
    colors = ["skyblue", "lightgreen", "lightcoral", "gold"]
    color_idx = 0

    if statistics["optimal_quality_values"]:
        quality_values, quality_counts = np.unique(list(statistics["optimal_quality_values"]), return_counts=True)
        ax.bar(
            [f"{q:.3f}" for q in quality_values],
            quality_counts,
            alpha=0.7,
            label="quality parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    if statistics["optimal_n_part_values"]:
        n_part_values, n_part_counts = np.unique(list(statistics["optimal_n_part_values"]), return_counts=True)
        ax.bar(
            [str(n) for n in n_part_values],
            n_part_counts,
            alpha=0.7,
            label="n_part parameter",
            color=colors[color_idx % len(colors)],
        )
        color_idx += 1

    if statistics["optimal_cfl_values"]:
        cfl_values, cfl_counts = np.unique(list(statistics["optimal_cfl_values"]), return_counts=True)
        ax.bar(
            [f"{c:.4f}" for c in cfl_values],
            cfl_counts,
            alpha=0.7,
            label="cfl parameter",
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
    plt.savefig(os.path.join(output_dir, "unstruct_mpm_statistics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Create detailed statistics file
    stats_file = os.path.join(output_dir, "unstruct_mpm_statistics_summary.txt")
    with open(stats_file, "w") as f:
        f.write("=== Unstruct MPM Dummy Search Statistics Summary ===\n\n")

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
        if statistics["optimal_quality_values"]:
            quality_values, quality_counts = np.unique(list(statistics["optimal_quality_values"]), return_counts=True)
            f.write("   quality parameter (iterative):\n")
            for quality, count in zip(quality_values, quality_counts):
                f.write(f"     quality={quality:.3f}: {count} times\n")

        if statistics["optimal_n_part_values"]:
            n_part_values, n_part_counts = np.unique(list(statistics["optimal_n_part_values"]), return_counts=True)
            f.write("   n_part parameter (iterative):\n")
            for n_part, count in zip(n_part_values, n_part_counts):
                f.write(f"     n_part={n_part}: {count} times\n")

        if statistics["optimal_cfl_values"]:
            cfl_values, cfl_counts = np.unique(list(statistics["optimal_cfl_values"]), return_counts=True)
            f.write("   cfl parameter (iterative):\n")
            for cfl, count in zip(cfl_values, cfl_counts):
                f.write(f"     cfl={cfl:.4f}: {count} times\n")


def main():
    print("=== Unstruct MPM Dummy Solution Generation ===")
    print("Loading configuration from unstruct_mpm.yaml...")

    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "unstruct_mpm.yaml")
    config = load_config(config_path)
    print("✅ Configuration loaded successfully")

    # Extract configuration sections
    precision_configs = {}
    for name, info in config["precision_levels"].items():
        # Only process precision levels with numeric values (skip placeholders)
        if isinstance(info["energy_tolerance"], (int, float)) and isinstance(info["var_threshold"], (int, float)):
            precision_configs[name] = {
                "energy_tolerance": info["energy_tolerance"],
                "var_threshold": info["var_threshold"],
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
        "optimal_nx_values": [],
        "optimal_quality_values": [],
        "optimal_n_part_values": [],
        "optimal_cfl_values": [],
    }

    # Initialize task collection for datasets
    successful_tasks = []
    failed_tasks = []

    # Generate all task combinations
    for precision_name, precision_vals in precision_configs.items():
        print(f"\n--- Processing {precision_name.upper()} precision ---")

        for profile in profiles:
            print(f"  Profile: {profile}")
            
            # Get case from profile configuration
            case = get_case_from_profile(profile)
            print(f"    Case: {case}")

            for target_param, target_config in target_configs.items():
                print(f"    Target parameter: {target_param}")

                # Get all non-target parameter names and their value lists
                non_target_params = target_config["non_target_parameters"]
                param_names = list(non_target_params.keys())
                # For profile-dependent parameters (like nx), select the correct value for the current profile
                param_values = []
                for name in param_names:
                    values = non_target_params[name]
                    if name == "nx":
                        # Use the nx list for the current profile
                        param_values.append(values[profile])
                    else:
                        param_values.append(values)                

                # Generate all combinations using nested loops (cartesian product)
                for combination in itertools.product(*param_values):
                    # Build parameters dictionary
                    task_params = dict(zip(param_names, combination))

                    print(f"      Running {target_param} search with params: {task_params}")
                    # Call appropriate search function based on target parameter
                    # Fixed radii = 1.0 for all simulations
                    if target_param == "nx":
                        # Get profile-specific initial nx value
                        initial_nx = target_config["initial_values"][profile]
                        is_converged, best_param, cost_history, param_history = find_convergent_nx(
                            profile=profile,
                            nx=initial_nx,
                            n_part=task_params["n_part"],
                            cfl=task_params["cfl"],
                            energy_tolerance=precision_vals["energy_tolerance"],
                            var_threshold=precision_vals["var_threshold"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                            case=case
                        )
                        if best_param is not None:
                            statistics["optimal_nx_values"].append(best_param)
                            quality = calculate_quality(best_param, profile)
                            statistics["optimal_quality_values"].append(quality)

                    elif target_param == "n_part":
                        is_converged, best_param, cost_history, param_history = find_convergent_n_part(
                            profile=profile,
                            nx=task_params["nx"],
                            n_part=target_config["initial_value"],
                            cfl=task_params["cfl"],
                            energy_tolerance=precision_vals["energy_tolerance"],
                            var_threshold=precision_vals["var_threshold"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                            case=case
                        )
                        if best_param is not None:
                            statistics["optimal_n_part_values"].append(best_param)

                    elif target_param == "cfl":
                        is_converged, best_param, cost_history, param_history = find_convergent_cfl(
                            profile=profile,
                            nx=task_params["nx"],
                            n_part=task_params["n_part"],
                            cfl=target_config["initial_value"],
                            energy_tolerance=precision_vals["energy_tolerance"],
                            var_threshold=precision_vals["var_threshold"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                            case=case
                        )
                        if best_param is not None:
                            statistics["optimal_cfl_values"].append(best_param)

                    # Create task record for dataset
                    task_record = {
                        "solver": "unstruct_mpm",
                        "target_parameter": target_param,
                        "profile": profile,
                        "precision_config": {
                            "energy_tolerance": precision_vals["energy_tolerance"],
                            "var_threshold": precision_vals["var_threshold"]
                        },
                        "target_config": {
                            "initial_value": target_config.get("initial_values", {}).get(profile) if target_param == "nx" else target_config.get("initial_value"),
                            "multiplication_factor": target_config.get("multiplication_factor"),
                            "max_iteration_num": target_config.get("max_iteration_num"),
                            "search_range_min": target_config.get("search_range_min"),
                            "search_range_max": target_config.get("search_range_max"),
                            "search_range_slice_num": target_config.get("search_range_slice_num")
                        },
                        "non_target_parameters": task_params.copy(),
                        "results": {
                            "converged": is_converged,
                            "optimal_parameter_value": best_param,
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
            combinations_per_target *= len(values) if isinstance(values, list) else len(values[profile])
        expected_total += len(profiles) * combinations_per_target

    print(f"\nTask breakdown:")
    for target_param, target_config in target_configs.items():
        param_values = [target_config["non_target_parameters"][name] for name in target_config["non_target_parameters"]]
        combinations_per_target = 1
        for values in param_values:
            combinations_per_target *= len(values) if isinstance(values, list) else len(values[profile])
        tasks_for_param = len(profiles) * combinations_per_target
        print(f"  {target_param}: {len(profiles)} profiles × {combinations_per_target} combos = {tasks_for_param}")
    print(f"  Expected total per precision: {expected_total}")
    print(f"  Expected total across {len(precision_configs)} precisions: {expected_total * len(precision_configs)}")
    print(f"  Actual total: {statistics['total_tasks']}")


if __name__ == "__main__":
    main()
