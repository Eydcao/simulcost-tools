#!/usr/bin/env python3
"""
NS Channel 2D Dummy Solution Generation Script

This script generates dummy solutions for all parameter combinations defined in the
ns_channel_2d.yaml file. It uses direct Python function calls instead of subprocess
for better performance and to capture return statistics.
"""

import sys
import os
import time
import yaml
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkouts.config_utils import load_config, build_target_configs
from dummy_sols.ns_channel_2d import (
    grid_search_mesh_x, grid_search_mesh_y, grid_search_omega_u, 
    grid_search_omega_v, grid_search_omega_p, grid_search_diff_u_threshold,
    grid_search_diff_v_threshold, grid_search_res_iter_v_threshold
)


def get_boundary_condition(profile):
    """Get boundary condition from profile config file"""
    config_path = f"run_configs/ns_channel_2d/{profile}.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'boundary_condition' in config:
                return config['boundary_condition']
            else:
                raise ValueError(f"boundary_condition not found in config file {config_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_path} not found. Please ensure the profile '{profile}' exists.")


def save_datasets(successful_tasks, failed_tasks, output_dir):
    """Save successful and failed tasks as separate JSON datasets in subfolders"""
    # Create subfolders for successful and failed tasks
    success_dir = os.path.join(output_dir, "ns_channel_2d", "successful")
    failed_dir = os.path.join(output_dir, "ns_channel_2d", "failed")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    
    # Save successful tasks (overwrite existing file)
    success_file = os.path.join(success_dir, "tasks.json")
    with open(success_file, "w") as f:  # "w" mode overwrites existing file
        json.dump({
            "metadata": {
                "solver": "ns_channel_2d",
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
                "solver": "ns_channel_2d", 
                "description": "Failed to converge parameter optimization tasks",
                "total_tasks": len(failed_tasks)
            },
            "tasks": failed_tasks
        }, f, indent=2)
    
    print(f"✅ Saved {len(successful_tasks)} successful tasks to {success_file}")
    print(f"❌ Saved {len(failed_tasks)} failed tasks to {failed_file}")
    
    return success_file, failed_file


def plot_statistics(statistics, output_dir):
    """Generate comprehensive statistical plots matching the format of other solvers."""
    print("\nGenerating statistical plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots - larger to accommodate all parameters
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Convergence rates by target parameter
    ax1 = plt.subplot(3, 4, 1)
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
    ax2 = plt.subplot(3, 4, 2)
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
    ax3 = plt.subplot(3, 4, 3)
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
    ax4 = plt.subplot(3, 4, 4)
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
    
    # Plot 5: Optimal parameter frequency for mesh parameters
    ax5 = plt.subplot(3, 4, 5)
    mesh_data = []
    mesh_labels = []
    mesh_colors = ["lightsteelblue", "lightgreen"]
    
    if statistics["optimal_mesh_x_values"]:
        mesh_x_counter = Counter([int(val) for val in statistics["optimal_mesh_x_values"]])
        most_common_mesh_x = mesh_x_counter.most_common(5)
        if most_common_mesh_x:
            values, counts = zip(*most_common_mesh_x)
            mesh_data.append(counts)
            mesh_labels.append("mesh_x")
    
    if statistics["optimal_mesh_y_values"]:
        mesh_y_counter = Counter([int(val) for val in statistics["optimal_mesh_y_values"]])
        most_common_mesh_y = mesh_y_counter.most_common(5)
        if most_common_mesh_y:
            values, counts = zip(*most_common_mesh_y)
            mesh_data.append(counts)
            mesh_labels.append("mesh_y")
    
    if mesh_data:
        x = np.arange(len(mesh_data[0]))
        width = 0.35
        for i, (data, label) in enumerate(zip(mesh_data, mesh_labels)):
            ax5.bar(x + i*width, data, width, label=label, color=mesh_colors[i])
        ax5.set_title("Most Frequent Optimal Mesh Values")
        ax5.set_xlabel("Rank")
        ax5.set_ylabel("Frequency")
        ax5.set_xticks(x + width/2)
        ax5.set_xticklabels([f"#{i+1}" for i in range(len(mesh_data[0]))])
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No optimal mesh\nvalues found', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title("Optimal Mesh Values")
    
    # Plot 6: Optimal parameter frequency for relaxation factors
    ax6 = plt.subplot(3, 4, 6)
    omega_data = []
    omega_labels = []
    omega_colors = ["lightcoral", "lightblue", "lightyellow"]
    
    if statistics["optimal_omega_u_values"]:
        omega_u_counter = Counter([round(val, 2) for val in statistics["optimal_omega_u_values"]])
        most_common_omega_u = omega_u_counter.most_common(5)
        if most_common_omega_u:
            values, counts = zip(*most_common_omega_u)
            omega_data.append(counts)
            omega_labels.append("omega_u")
    
    if statistics["optimal_omega_v_values"]:
        omega_v_counter = Counter([round(val, 2) for val in statistics["optimal_omega_v_values"]])
        most_common_omega_v = omega_v_counter.most_common(5)
        if most_common_omega_v:
            values, counts = zip(*most_common_omega_v)
            omega_data.append(counts)
            omega_labels.append("omega_v")
    
    if statistics["optimal_omega_p_values"]:
        omega_p_counter = Counter([round(val, 2) for val in statistics["optimal_omega_p_values"]])
        most_common_omega_p = omega_p_counter.most_common(5)
        if most_common_omega_p:
            values, counts = zip(*most_common_omega_p)
            omega_data.append(counts)
            omega_labels.append("omega_p")
    
    if omega_data:
        x = np.arange(len(omega_data[0]))
        width = 0.25
        for i, (data, label) in enumerate(zip(omega_data, omega_labels)):
            ax6.bar(x + i*width, data, width, label=label, color=omega_colors[i])
        ax6.set_title("Most Frequent Optimal Relaxation Factors")
        ax6.set_xlabel("Rank")
        ax6.set_ylabel("Frequency")
        ax6.set_xticks(x + width)
        ax6.set_xticklabels([f"#{i+1}" for i in range(len(omega_data[0]))])
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'No optimal relaxation\nfactor values found', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title("Optimal Relaxation Factors")
    
    # Plot 7: Optimal parameter frequency for diff_u_threshold
    ax7 = plt.subplot(3, 4, 7)
    if statistics["optimal_diff_u_threshold_values"]:
        diff_u_counter = Counter([round(val, 8) for val in statistics["optimal_diff_u_threshold_values"]])
        most_common_diff_u = diff_u_counter.most_common(5)
        
        if most_common_diff_u:
            values, counts = zip(*most_common_diff_u)
            bars7 = ax7.bar(range(len(values)), counts, color="lightpink")
            ax7.set_title("Most Frequent Optimal diff_u_threshold")
            ax7.set_xlabel("Threshold Value")
            ax7.set_ylabel("Frequency")
            ax7.set_xticks(range(len(values)))
            ax7.set_xticklabels([f"{v:.1e}" for v in values], rotation=45, ha='right')
            
            for bar, count in zip(bars7, counts):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                        str(count), ha='center', va='bottom')
    else:
        ax7.text(0.5, 0.5, 'No optimal diff_u_threshold\nvalues found', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title("Optimal diff_u_threshold Values")
    
    # Plot 8: Optimal parameter frequency for diff_v_threshold
    ax8 = plt.subplot(3, 4, 8)
    if statistics["optimal_diff_v_threshold_values"]:
        diff_v_counter = Counter([round(val, 8) for val in statistics["optimal_diff_v_threshold_values"]])
        most_common_diff_v = diff_v_counter.most_common(5)
        
        if most_common_diff_v:
            values, counts = zip(*most_common_diff_v)
            bars8 = ax8.bar(range(len(values)), counts, color="lightcyan")
            ax8.set_title("Most Frequent Optimal diff_v_threshold")
            ax8.set_xlabel("Threshold Value")
            ax8.set_ylabel("Frequency")
            ax8.set_xticks(range(len(values)))
            ax8.set_xticklabels([f"{v:.1e}" for v in values], rotation=45, ha='right')
            
            for bar, count in zip(bars8, counts):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                        str(count), ha='center', va='bottom')
    else:
        ax8.text(0.5, 0.5, 'No optimal diff_v_threshold\nvalues found', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=12)
        ax8.set_title("Optimal diff_v_threshold Values")
    
    # Plot 9: Optimal parameter frequency for res_iter_v_threshold
    ax9 = plt.subplot(3, 4, 9)
    if statistics["optimal_res_iter_v_threshold_values"]:
        res_iter_counter = Counter([round(val, 8) for val in statistics["optimal_res_iter_v_threshold_values"]])
        most_common_res_iter = res_iter_counter.most_common(5)
        
        if most_common_res_iter:
            values, counts = zip(*most_common_res_iter)
            bars9 = ax9.bar(range(len(values)), counts, color="lightyellow")
            ax9.set_title("Most Frequent Optimal res_iter_v_threshold")
            ax9.set_xlabel("Threshold Value")
            ax9.set_ylabel("Frequency")
            ax9.set_xticks(range(len(values)))
            ax9.set_xticklabels([f"{v:.1e}" for v in values], rotation=45, ha='right')
            
            for bar, count in zip(bars9, counts):
                height = bar.get_height()
                ax9.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                        str(count), ha='center', va='bottom')
    else:
        ax9.text(0.5, 0.5, 'No optimal res_iter_v_threshold\nvalues found', 
                ha='center', va='center', transform=ax9.transAxes, fontsize=12)
        ax9.set_title("Optimal res_iter_v_threshold Values")
    
    # Plot 10: Parameter convergence comparison
    ax10 = plt.subplot(3, 4, 10)
    param_names = list(statistics["convergence_by_target"].keys())
    convergence_counts = []
    for param in param_names:
        converged = statistics["convergence_by_target"][param]["converged"]
        total = statistics["convergence_by_target"][param]["total"]
        convergence_counts.append(converged)
    
    bars10 = ax10.bar(param_names, convergence_counts, color=["skyblue", "lightgreen", "lightcoral", "gold", "lightpink", "lightcyan", "lightyellow"])
    ax10.set_title("Convergence Count by Parameter")
    ax10.set_ylabel("Number of Converged Tasks")
    ax10.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars10, convergence_counts):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                str(count), ha='center', va='bottom')
    
    # Plot 11: Average cost by parameter
    ax11 = plt.subplot(3, 4, 11)
    avg_costs = []
    for param in param_names:
        costs = statistics["convergence_by_target"][param]["costs"]
        avg_cost = np.mean(costs) if costs else 0
        avg_costs.append(avg_cost)
    
    bars11 = ax11.bar(param_names, avg_costs, color=["skyblue", "lightgreen", "lightcoral", "gold", "lightpink", "lightcyan", "lightyellow"])
    ax11.set_title("Average Cost by Parameter")
    ax11.set_ylabel("Average Computational Cost")
    ax11.tick_params(axis='x', rotation=45)
    ax11.set_yscale('log')
    
    for bar, cost in zip(bars11, avg_costs):
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2., height * 1.1, 
                f"{cost:.0f}", ha='center', va='bottom', fontsize=8)
    
    # Plot 12: Parameter distribution summary
    ax12 = plt.subplot(3, 4, 12)
    # Create a summary table showing parameter statistics
    summary_data = []
    for param_name, values in [("mesh_x", statistics["optimal_mesh_x_values"]), 
                               ("mesh_y", statistics["optimal_mesh_y_values"]),
                               ("omega_u", statistics["optimal_omega_u_values"]),
                               ("omega_v", statistics["optimal_omega_v_values"]),
                               ("omega_p", statistics["optimal_omega_p_values"]),
                               ("diff_u", statistics["optimal_diff_u_threshold_values"]),
                               ("diff_v", statistics["optimal_diff_v_threshold_values"]),
                               ("res_iter", statistics["optimal_res_iter_v_threshold_values"])]:
        if values:
            if param_name in ["mesh_x", "mesh_y"]:
                counter = Counter([int(val) for val in values])
            elif param_name in ["omega_u", "omega_v", "omega_p"]:
                counter = Counter([round(val, 2) for val in values])
            else:
                counter = Counter([round(val, 8) for val in values])
                
            most_common = counter.most_common(1)[0]
            frequency = most_common[1] / len(values) * 100
            summary_data.append([param_name, most_common[0], f"{frequency:.1f}%"])
        else:
            summary_data.append([param_name, "N/A", "0%"])
    
    # Create table
    table = ax12.table(cellText=summary_data, 
                      colLabels=["Parameter", "Most Common", "Frequency"],
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    ax12.set_title("Parameter Distribution Summary")
    ax12.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ns_channel_2d_statistics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary text file
    with open(os.path.join(output_dir, "ns_channel_2d_statistics_summary.txt"), "w") as f:
        f.write("NS CHANNEL 2D DUMMY SOLUTION STATISTICS SUMMARY\n")
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
        for param_name, values in [("mesh_x", statistics["optimal_mesh_x_values"]), 
                                   ("mesh_y", statistics["optimal_mesh_y_values"]),
                                   ("omega_u", statistics["optimal_omega_u_values"]),
                                   ("omega_v", statistics["optimal_omega_v_values"]),
                                   ("omega_p", statistics["optimal_omega_p_values"]),
                                   ("diff_u_threshold", statistics["optimal_diff_u_threshold_values"]),
                                   ("diff_v_threshold", statistics["optimal_diff_v_threshold_values"]),
                                   ("res_iter_v_threshold", statistics["optimal_res_iter_v_threshold_values"])]:
            if values:
                if param_name in ["mesh_x", "mesh_y"]:
                    counter = Counter([int(val) for val in values])
                elif param_name in ["omega_u", "omega_v", "omega_p"]:
                    counter = Counter([round(val, 2) for val in values])
                else:
                    counter = Counter([round(val, 6) for val in values])
                    
                most_common = counter.most_common(1)[0]
                frequency = most_common[1] / len(values) * 100
                f.write(f"  {param_name}: {most_common[0]} (appears {frequency:.1f}% of the time)\n")
            else:
                f.write(f"  {param_name}: No values found\n")


def main():
    """Main execution function."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("NS CHANNEL 2D DUMMY SOLUTION GENERATION")
    print("=" * 60)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "ns_channel_2d.yaml")
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
        "optimal_mesh_x_values": [],
        "optimal_mesh_y_values": [],
        "optimal_omega_u_values": [],
        "optimal_omega_v_values": [],
        "optimal_omega_p_values": [],
        "optimal_diff_u_threshold_values": [],
        "optimal_diff_v_threshold_values": [],
        "optimal_res_iter_v_threshold_values": []
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
                    if target_param == "mesh_x":
                        # For mesh_x, we need to get the mesh_x values from the config
                        if "candidates_by_precision" in target_config and precision_name in target_config["candidates_by_precision"]:
                            mesh_x_values = target_config["candidates_by_precision"][precision_name]["mesh_x"]
                        else:
                            # Fallback to typical values or generate range
                            mesh_x_values = [96, 128, 160, 192, 256, 320, 384, 448, 512]
                        boundary_condition = get_boundary_condition(profile)
                        is_converged, best_param, cost_history, param_history = grid_search_mesh_x(
                            profile=profile,
                            boundary_condition=boundary_condition,
                            mesh_x_values=mesh_x_values,
                            mesh_y=task_params["mesh_y"],
                            omega_u=task_params["omega_u"],
                            omega_v=task_params["omega_v"],
                            omega_p=task_params["omega_p"],
                            diff_u_threshold=task_params["diff_u_threshold"],
                            diff_v_threshold=task_params["diff_v_threshold"],
                            res_iter_v_threshold=task_params["res_iter_v_threshold"],
                            length=20.0,  # Default from config
                            breadth=1.0,  # Default from config
                            mass_tolerance=precision_vals["mass_tolerance"],
                            u_rmse_tolerance=precision_vals["u_rmse_tolerance"],
                            v_rmse_tolerance=precision_vals["v_rmse_tolerance"],
                            p_rmse_tolerance=precision_vals["p_rmse_tolerance"]
                        )
                        if best_param is not None:
                            statistics["optimal_mesh_x_values"].append(best_param)

                    elif target_param == "mesh_y":
                        # For mesh_y, we need to get the mesh_y values from the config
                        if "candidates_by_precision" in target_config and precision_name in target_config["candidates_by_precision"]:
                            mesh_y_values = target_config["candidates_by_precision"][precision_name]["mesh_y"]
                        else:
                            # Fallback to typical values or generate range
                            mesh_y_values = [24, 32, 40, 48, 64, 80, 96, 112, 128]
                        boundary_condition = get_boundary_condition(profile)
                        is_converged, best_param, cost_history, param_history = grid_search_mesh_y(
                            profile=profile,
                            boundary_condition=boundary_condition,
                            mesh_x=task_params["mesh_x"],
                            mesh_y_values=mesh_y_values,
                            omega_u=task_params["omega_u"],
                            omega_v=task_params["omega_v"],
                            omega_p=task_params["omega_p"],
                            diff_u_threshold=task_params["diff_u_threshold"],
                            diff_v_threshold=task_params["diff_v_threshold"],
                            res_iter_v_threshold=task_params["res_iter_v_threshold"],
                            length=20.0,  # Default from config
                            breadth=1.0,  # Default from config
                            mass_tolerance=precision_vals["mass_tolerance"],
                            u_rmse_tolerance=precision_vals["u_rmse_tolerance"],
                            v_rmse_tolerance=precision_vals["v_rmse_tolerance"],
                            p_rmse_tolerance=precision_vals["p_rmse_tolerance"]
                        )
                        if best_param is not None:
                            statistics["optimal_mesh_y_values"].append(best_param)

                    elif target_param == "omega_u":
                        # For omega_u, use typical values from config or generate range
                        if "typical_values" in target_config:
                            omega_u_values = target_config["typical_values"]
                        else:
                            omega_u_values = [0.1 * i for i in range(1, 11)]  # 0.1 to 1.0
                        boundary_condition = get_boundary_condition(profile)
                        is_converged, best_param, cost_history, param_history = grid_search_omega_u(
                            profile=profile,
                            boundary_condition=boundary_condition,
                            mesh_x=task_params["mesh_x"],
                            mesh_y=task_params["mesh_y"],
                            omega_u_values=omega_u_values,
                            omega_v=task_params["omega_v"],
                            omega_p=task_params["omega_p"],
                            diff_u_threshold=task_params["diff_u_threshold"],
                            diff_v_threshold=task_params["diff_v_threshold"],
                            res_iter_v_threshold=task_params["res_iter_v_threshold"],
                            length=20.0,  # Default from config
                            breadth=1.0,  # Default from config
                            mass_tolerance=precision_vals["mass_tolerance"],
                            u_rmse_tolerance=precision_vals["u_rmse_tolerance"],
                            v_rmse_tolerance=precision_vals["v_rmse_tolerance"],
                            p_rmse_tolerance=precision_vals["p_rmse_tolerance"]
                        )
                        if best_param is not None:
                            statistics["optimal_omega_u_values"].append(best_param)

                    elif target_param == "omega_v":
                        # For omega_v, use typical values from config or generate range
                        if "typical_values" in target_config:
                            omega_v_values = target_config["typical_values"]
                        else:
                            omega_v_values = [0.1 * i for i in range(1, 11)]  # 0.1 to 1.0
                        boundary_condition = get_boundary_condition(profile)
                        is_converged, best_param, cost_history, param_history = grid_search_omega_v(
                            profile=profile,
                            boundary_condition=boundary_condition,
                            mesh_x=task_params["mesh_x"],
                            mesh_y=task_params["mesh_y"],
                            omega_u=task_params["omega_u"],
                            omega_v_values=omega_v_values,
                            omega_p=task_params["omega_p"],
                            diff_u_threshold=task_params["diff_u_threshold"],
                            diff_v_threshold=task_params["diff_v_threshold"],
                            res_iter_v_threshold=task_params["res_iter_v_threshold"],
                            length=20.0,  # Default from config
                            breadth=1.0,  # Default from config
                            mass_tolerance=precision_vals["mass_tolerance"],
                            u_rmse_tolerance=precision_vals["u_rmse_tolerance"],
                            v_rmse_tolerance=precision_vals["v_rmse_tolerance"],
                            p_rmse_tolerance=precision_vals["p_rmse_tolerance"]
                        )
                        if best_param is not None:
                            statistics["optimal_omega_v_values"].append(best_param)

                    elif target_param == "omega_p":
                        # For omega_p, use typical values from config or generate range
                        if "typical_values" in target_config:
                            omega_p_values = target_config["typical_values"]
                        else:
                            omega_p_values = [0.1 * i for i in range(1, 6)]  # 0.1 to 0.5
                        boundary_condition = get_boundary_condition(profile)
                        is_converged, best_param, cost_history, param_history = grid_search_omega_p(
                            profile=profile,
                            boundary_condition=boundary_condition,
                            mesh_x=task_params["mesh_x"],
                            mesh_y=task_params["mesh_y"],
                            omega_u=task_params["omega_u"],
                            omega_v=task_params["omega_v"],
                            omega_p_values=omega_p_values,
                            diff_u_threshold=task_params["diff_u_threshold"],
                            diff_v_threshold=task_params["diff_v_threshold"],
                            res_iter_v_threshold=task_params["res_iter_v_threshold"],
                            length=20.0,  # Default from config
                            breadth=1.0,  # Default from config
                            mass_tolerance=precision_vals["mass_tolerance"],
                            u_rmse_tolerance=precision_vals["u_rmse_tolerance"],
                            v_rmse_tolerance=precision_vals["v_rmse_tolerance"],
                            p_rmse_tolerance=precision_vals["p_rmse_tolerance"]
                        )
                        if best_param is not None:
                            statistics["optimal_omega_p_values"].append(best_param)

                    elif target_param == "diff_u_threshold":
                        # For diff_u_threshold, use values from config
                        if "diff_candidates_by_precision" in target_config:
                            diff_u_values = target_config["diff_candidates_by_precision"][precision_name]["diff_u_threshold"]
                        elif "typical_values" in target_config:
                            diff_u_values = target_config["typical_values"]
                        else:
                            diff_u_values = [1e-6, 1e-7, 1e-8]
                        boundary_condition = get_boundary_condition(profile)
                        is_converged, best_param, cost_history, param_history = grid_search_diff_u_threshold(
                            profile=profile,
                            boundary_condition=boundary_condition,
                            mesh_x=task_params["mesh_x"],
                            mesh_y=task_params["mesh_y"],
                            omega_u=task_params["omega_u"],
                            omega_v=task_params["omega_v"],
                            omega_p=task_params["omega_p"],
                            diff_u_values=diff_u_values,
                            diff_v_threshold=task_params["diff_v_threshold"],
                            res_iter_v_threshold=task_params["res_iter_v_threshold"],
                            length=20.0,  # Default from config
                            breadth=1.0,  # Default from config
                            mass_tolerance=precision_vals["mass_tolerance"],
                            u_rmse_tolerance=precision_vals["u_rmse_tolerance"],
                            v_rmse_tolerance=precision_vals["v_rmse_tolerance"],
                            p_rmse_tolerance=precision_vals["p_rmse_tolerance"]
                        )
                        if best_param is not None:
                            statistics["optimal_diff_u_threshold_values"].append(best_param)

                    elif target_param == "diff_v_threshold":
                        # For diff_v_threshold, use values from config
                        if "diff_candidates_by_precision" in target_config:
                            diff_v_values = target_config["diff_candidates_by_precision"][precision_name]["diff_v_threshold"]
                        elif "typical_values" in target_config:
                            diff_v_values = target_config["typical_values"]
                        else:
                            diff_v_values = [1e-6, 1e-7, 1e-8]
                        boundary_condition = get_boundary_condition(profile)
                        is_converged, best_param, cost_history, param_history = grid_search_diff_v_threshold(
                            profile=profile,
                            boundary_condition=boundary_condition,
                            mesh_x=task_params["mesh_x"],
                            mesh_y=task_params["mesh_y"],
                            omega_u=task_params["omega_u"],
                            omega_v=task_params["omega_v"],
                            omega_p=task_params["omega_p"],
                            diff_u_threshold=task_params["diff_u_threshold"],
                            diff_v_values=diff_v_values,
                            res_iter_v_threshold=task_params["res_iter_v_threshold"],
                            length=20.0,  # Default from config
                            breadth=1.0,  # Default from config
                            mass_tolerance=precision_vals["mass_tolerance"],
                            u_rmse_tolerance=precision_vals["u_rmse_tolerance"],
                            v_rmse_tolerance=precision_vals["v_rmse_tolerance"],
                            p_rmse_tolerance=precision_vals["p_rmse_tolerance"]
                        )
                        if best_param is not None:
                            statistics["optimal_diff_v_threshold_values"].append(best_param)

                    elif target_param == "res_iter_v_threshold":
                        # For res_iter_v_threshold, use values from config
                        if "schedule_options" in target_config:
                            res_iter_v_values = target_config["schedule_options"]
                        elif "typical_values" in target_config:
                            res_iter_v_values = target_config["typical_values"]
                        else:
                            res_iter_v_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                        boundary_condition = get_boundary_condition(profile)
                        is_converged, best_param, cost_history, param_history = grid_search_res_iter_v_threshold(
                            profile=profile,
                            boundary_condition=boundary_condition,
                            mesh_x=task_params["mesh_x"],
                            mesh_y=task_params["mesh_y"],
                            omega_u=task_params["omega_u"],
                            omega_v=task_params["omega_v"],
                            omega_p=task_params["omega_p"],
                            diff_u_threshold=task_params["diff_u_threshold"],
                            diff_v_threshold=task_params["diff_v_threshold"],
                            res_iter_v_values=res_iter_v_values,
                            length=20.0,  # Default from config
                            breadth=1.0,  # Default from config
                            mass_tolerance=precision_vals["mass_tolerance"],
                            u_rmse_tolerance=precision_vals["u_rmse_tolerance"],
                            v_rmse_tolerance=precision_vals["v_rmse_tolerance"],
                            p_rmse_tolerance=precision_vals["p_rmse_tolerance"]
                        )
                        if best_param is not None:
                            statistics["optimal_res_iter_v_threshold_values"].append(best_param)

                    # Create task record for dataset
                    task_record = {
                        "solver": "ns_channel_2d",
                        "target_parameter": target_param,
                        "profile": profile,
                        "precision_level": precision_name,
                        "precision_config": {
                            "mass_tolerance": precision_vals["mass_tolerance"],
                            "u_rmse_tolerance": precision_vals["u_rmse_tolerance"],
                            "v_rmse_tolerance": precision_vals["v_rmse_tolerance"],
                            "p_rmse_tolerance": precision_vals["p_rmse_tolerance"],
                            "max_iter": precision_vals["max_iter"]
                        },
                        "target_config": {
                            "initial_value": target_config.get("initial_value"),
                            "initial_values": target_config.get("initial_values"),
                            "candidates_by_precision": target_config.get("candidates_by_precision"),
                            "diff_candidates_by_precision": target_config.get("diff_candidates_by_precision"),
                            "typical_values": target_config.get("typical_values"),
                            "schedule_options": target_config.get("schedule_options"),
                            "max_iteration_num": target_config.get("max_iteration_num"),
                            "search_range": target_config.get("search_range"),
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
    print("NS CHANNEL 2D DUMMY SOLUTION GENERATION COMPLETED!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
