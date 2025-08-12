#!/usr/bin/env python3
"""
Burgers 1D Dummy Solution Generation Script

This script generates dummy solutions for all parameter combinations defined in the
burgers_1d_config.yaml file. It uses direct Python function calls instead of subprocess
for better performance and to capture return statistics.
"""

import sys
import os
import time
import yaml
import json
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkouts.config_utils import load_config, build_target_configs
from dummy_sols.burgers_1d import find_convergent_cfl, find_convergent_n_space, find_optimal_k, find_optimal_w


def save_datasets(successful_tasks, failed_tasks, output_dir):
    """Save successful and failed tasks as separate JSON datasets in subfolders"""
    # Create subfolders for successful and failed tasks
    success_dir = os.path.join(output_dir, "burgers_1d", "successful")
    failed_dir = os.path.join(output_dir, "burgers_1d", "failed")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    
    # Save successful tasks (overwrite existing file)
    success_file = os.path.join(success_dir, "tasks.json")
    with open(success_file, "w") as f:  # "w" mode overwrites existing file
        json.dump({
            "metadata": {
                "solver": "burgers_1d",
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
                "solver": "burgers_1d", 
                "description": "Failed to converge parameter optimization tasks",
                "total_tasks": len(failed_tasks)
            },
            "tasks": failed_tasks
        }, f, indent=2)
    
    print(f"✅ Saved {len(successful_tasks)} successful tasks to {success_file}")
    print(f"❌ Saved {len(failed_tasks)} failed tasks to {failed_file}")
    
    return success_file, failed_file




def plot_statistics(statistics, output_dir):
    """Generate comprehensive statistical plots matching Euler 1D format."""
    print("\nGenerating statistical plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Convergence rates by target parameter
    ax1 = plt.subplot(2, 3, 1)
    target_params = list(statistics["convergence_by_target"].keys())
    convergence_rates = []
    for param in target_params:
        total = statistics["convergence_by_target"][param]["total"]
        converged = statistics["convergence_by_target"][param]["converged"]
        rate = converged / max(total, 1)
        convergence_rates.append(rate)
    
    bars1 = ax1.bar(target_params, convergence_rates, color=["skyblue", "lightgreen", "lightcoral", "gold"])
    ax1.set_title("Convergence Rate by Target Parameter")
    ax1.set_ylabel("Convergence Rate")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for bar, rate in zip(bars1, convergence_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Convergence rates by precision level
    ax2 = plt.subplot(2, 3, 2)
    precision_levels = list(statistics["convergence_by_precision"].keys())
    precision_rates = []
    for precision in precision_levels:
        total = statistics["convergence_by_precision"][precision]["total"]
        converged = statistics["convergence_by_precision"][precision]["converged"]
        rate = converged / max(total, 1)
        precision_rates.append(rate)
    
    bars2 = ax2.bar(precision_levels, precision_rates, color=["lightblue", "orange", "lightgreen"])
    ax2.set_title("Convergence Rate by Precision Level")
    ax2.set_ylabel("Convergence Rate")
    ax2.set_ylim(0, 1)
    
    for bar, rate in zip(bars2, precision_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Convergence rates by profile
    ax3 = plt.subplot(2, 3, 3)
    profiles = list(statistics["convergence_by_profile"].keys())
    profile_rates = []
    for profile in profiles:
        total = statistics["convergence_by_profile"][profile]["total"]
        converged = statistics["convergence_by_profile"][profile]["converged"]
        rate = converged / max(total, 1)
        profile_rates.append(rate)
    
    bars3 = ax3.bar(profiles, profile_rates, color=["pink", "lightcyan", "wheat", "lavender", "lightsalmon"])
    ax3.set_title("Convergence Rate by Profile")
    ax3.set_ylabel("Convergence Rate")
    ax3.set_ylim(0, 1)
    
    for bar, rate in zip(bars3, profile_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Cost distribution by target parameter
    ax4 = plt.subplot(2, 3, 4)
    cost_data = []
    cost_labels = []
    for param in target_params:
        costs = statistics["convergence_by_target"][param]["costs"]
        if costs:
            cost_data.append(costs)
            cost_labels.append(param)
    
    if cost_data:
        ax4.boxplot(cost_data, labels=cost_labels)
        ax4.set_title("Computational Cost Distribution")
        ax4.set_ylabel("Total Cost")
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_yscale('log')
    
    # Plot 5: Optimal parameter frequency for CFL
    ax5 = plt.subplot(2, 3, 5)
    if statistics["optimal_cfl_values"]:
        from collections import Counter
        cfl_counter = Counter([round(val, 4) for val in statistics["optimal_cfl_values"]])
        most_common_cfl = cfl_counter.most_common(10)
        
        if most_common_cfl:
            values, counts = zip(*most_common_cfl)
            bars5 = ax5.bar(range(len(values)), counts, color="lightsteelblue")
            ax5.set_title("Most Frequent Optimal CFL Values")
            ax5.set_xlabel("CFL Value")
            ax5.set_ylabel("Frequency")
            ax5.set_xticks(range(len(values)))
            ax5.set_xticklabels([str(v) for v in values], rotation=45, ha='right')
            
            for bar, count in zip(bars5, counts):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                        str(count), ha='center', va='bottom')
    else:
        ax5.text(0.5, 0.5, 'No optimal CFL\nvalues found', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title("Optimal CFL Values")
    
    # Plot 6: Optimal parameter frequency for k
    ax6 = plt.subplot(2, 3, 6)
    if statistics["optimal_k_values"]:
        from collections import Counter
        k_counter = Counter([round(val, 2) for val in statistics["optimal_k_values"]])
        most_common_k = k_counter.most_common(10)
        
        if most_common_k:
            values, counts = zip(*most_common_k)
            bars6 = ax6.bar(range(len(values)), counts, color="lightcoral")
            ax6.set_title("Most Frequent Optimal k Values")
            ax6.set_xlabel("k Value")
            ax6.set_ylabel("Frequency")
            ax6.set_xticks(range(len(values)))
            ax6.set_xticklabels([str(v) for v in values], rotation=45, ha='right')
            
            for bar, count in zip(bars6, counts):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                        str(count), ha='center', va='bottom')
    else:
        ax6.text(0.5, 0.5, 'No optimal k\nvalues found', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title("Optimal k Values")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "burgers_1d_statistics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary text file
    with open(os.path.join(output_dir, "burgers_1d_statistics_summary.txt"), "w") as f:
        f.write("BURGERS 1D DUMMY SOLUTION STATISTICS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total tasks executed: {statistics['total_tasks']}\n")
        f.write(f"Successfully converged: {statistics['total_converged']}\n")
        f.write(f"Overall convergence rate: {(statistics['total_converged']/max(statistics['total_tasks'], 1)*100):.2f}%\n\n")
        
        f.write("CONVERGENCE BY TARGET PARAMETER:\n")
        for param, stats in statistics["convergence_by_target"].items():
            rate = stats["converged"] / max(stats["total"], 1) * 100
            f.write(f"  {param.upper()}: {rate:.2f}% ({stats['converged']}/{stats['total']})\n")
        
        f.write("\nCONVERGENCE BY PRECISION LEVEL:\n")
        for precision, stats in statistics["convergence_by_precision"].items():
            rate = stats["converged"] / max(stats["total"], 1) * 100
            f.write(f"  {precision.upper()}: {rate:.2f}% ({stats['converged']}/{stats['total']})\n")
        
        f.write("\nCONVERGENCE BY PROFILE:\n")
        for profile, stats in statistics["convergence_by_profile"].items():
            rate = stats["converged"] / max(stats["total"], 1) * 100
            f.write(f"  {profile.upper()}: {rate:.2f}% ({stats['converged']}/{stats['total']})\n")
        
        f.write(f"\nMOST FREQUENT OPTIMAL VALUES:\n")
        for param_name, values in [("CFL", statistics["optimal_cfl_values"]), 
                                   ("k", statistics["optimal_k_values"]),
                                   ("w", statistics["optimal_w_values"]),
                                   ("n_space", statistics["optimal_n_space_values"])]:
            if values:
                from collections import Counter
                if param_name == "CFL":
                    counter = Counter([round(val, 4) for val in values])
                elif param_name in ["k", "w"]:
                    counter = Counter([round(val, 2) for val in values])
                else:
                    counter = Counter([int(val) for val in values])
                    
                most_common = counter.most_common(1)[0]
                frequency = most_common[1] / len(values) * 100
                f.write(f"  {param_name}: {most_common[0]} (appears {frequency:.1f}% of the time)\n")
            else:
                f.write(f"  {param_name}: No values found\n")


def main():
    """Main execution function."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("BURGERS 1D DUMMY SOLUTION GENERATION")
    print("=" * 60)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "burgers_1d.yaml")
    config = load_config(config_path)
    target_configs = build_target_configs(config)
    precision_configs = config["precision_levels"]
    profiles = config["profiles"]["active_profiles"]
    
    # Print configuration overview
    print(f"Profiles: {profiles}")
    print(f"Target parameters: {list(target_configs.keys())}")
    print(f"Precision levels: {list(precision_configs.keys())}")
    
    # Initialize statistics tracking
    statistics = {
        "total_tasks": 0,
        "total_converged": 0,
        "convergence_by_precision": {},
        "convergence_by_target": {},
        "convergence_by_profile": {},
        "optimal_cfl_values": [],
        "optimal_n_space_values": [],
        "optimal_k_values": [], 
        "optimal_w_values": []
    }
    
    # Initialize nested dictionaries
    for precision_name in precision_configs.keys():
        statistics["convergence_by_precision"][precision_name] = {"converged": 0, "total": 0}
    for target_param in target_configs.keys():
        statistics["convergence_by_target"][target_param] = {"converged": 0, "total": 0, "costs": []}
    for profile in profiles:
        statistics["convergence_by_profile"][profile] = {"converged": 0, "total": 0}
    
    # Data collection for datasets
    successful_tasks = []
    failed_tasks = []

    # Generate all task combinations
    import itertools
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
                    if target_param == "cfl":
                        is_converged, best_param, cost_history, param_history = find_convergent_cfl(
                            profile=profile,
                            cfl=target_config["initial_value"],
                            k=task_params["k"],
                            w=task_params["w"],
                            tolerance_infity=precision_vals["tolerance_linf"],
                            tolerance_2=precision_vals["tolerance_rmse"],
                        )
                        if best_param is not None:
                            statistics["optimal_cfl_values"].append(best_param)

                    elif target_param == "n_space":
                        is_converged, best_param, cost_history, param_history = find_convergent_n_space(
                            profile=profile,
                            n_space=target_config["initial_value"],
                            cfl=task_params["cfl"],
                            k=task_params["k"],
                            w=task_params["w"],
                            tolerance_infity=precision_vals["tolerance_linf"],
                            tolerance_2=precision_vals["tolerance_rmse"],
                        )
                        if best_param is not None:
                            statistics["optimal_n_space_values"].append(best_param)

                    elif target_param == "k":
                        is_converged, optimal_params, optimal_cost_history, param_history = find_optimal_k(
                            profile=profile, 
                            w=task_params["w"], 
                            tolerance_infity=precision_vals["tolerance_linf"], 
                            tolerance_2=precision_vals["tolerance_rmse"]
                        )
                        best_param, optimal_cfl = optimal_params if optimal_params != (None, None) else (None, None)
                        cost_history = optimal_cost_history
                        if best_param is not None:
                            statistics["optimal_k_values"].append(best_param)

                    elif target_param == "w":
                        is_converged, optimal_params, optimal_cost_history, param_history = find_optimal_w(
                            profile=profile, 
                            k=task_params["k"], 
                            tolerance_infity=precision_vals["tolerance_linf"], 
                            tolerance_2=precision_vals["tolerance_rmse"]
                        )
                        best_param, optimal_cfl = optimal_params if optimal_params != (None, None) else (None, None)
                        cost_history = optimal_cost_history
                        if best_param is not None:
                            statistics["optimal_w_values"].append(best_param)

                    # Create task record for dataset
                    task_record = {
                        "solver": "burgers_1d",
                        "target_parameter": target_param,
                        "profile": profile,
                        "precision_level": precision_name,
                        "precision_config": {
                            "tolerance_linf": precision_vals["tolerance_linf"],
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
    
    print(f"\n{'='*60}")
    print("BURGERS 1D DUMMY SOLUTION GENERATION COMPLETED!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
