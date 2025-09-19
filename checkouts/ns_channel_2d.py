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
        # Find the maximum length to ensure all arrays have the same shape
        max_len = max(len(data) for data in mesh_data)
        x = np.arange(max_len)
        width = 0.35
        for i, (data, label) in enumerate(zip(mesh_data, mesh_labels)):
            # Pad the data array to match the maximum length
            padded_data = list(data) + [0] * (max_len - len(data))
            ax5.bar(x + i*width, padded_data, width, label=label, color=mesh_colors[i])
        ax5.set_title("Most Frequent Optimal Mesh Values (Successful Tasks Only)")
        ax5.set_xlabel("Rank")
        ax5.set_ylabel("Frequency")
        ax5.set_xticks(x + width/2)
        ax5.set_xticklabels([f"#{i+1}" for i in range(max_len)])
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No optimal mesh\nvalues found', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title("Optimal Mesh Values (Successful Tasks Only)")
    
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
        # Find the maximum length to ensure all arrays have the same shape
        max_len = max(len(data) for data in omega_data)
        x = np.arange(max_len)
        width = 0.25
        for i, (data, label) in enumerate(zip(omega_data, omega_labels)):
            # Pad the data array to match the maximum length
            padded_data = list(data) + [0] * (max_len - len(data))
            ax6.bar(x + i*width, padded_data, width, label=label, color=omega_colors[i])
        ax6.set_title("Most Frequent Optimal Relaxation Factors (Successful Tasks Only)")
        ax6.set_xlabel("Rank")
        ax6.set_ylabel("Frequency")
        ax6.set_xticks(x + width)
        ax6.set_xticklabels([f"#{i+1}" for i in range(max_len)])
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'No optimal relaxation\nfactor values found', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title("Optimal Relaxation Factors (Successful Tasks Only)")
    
    # Plot 7: Optimal parameter frequency for diff_u_threshold
    ax7 = plt.subplot(3, 4, 7)
    if statistics["optimal_diff_u_threshold_values"]:
        diff_u_counter = Counter([round(val, 8) for val in statistics["optimal_diff_u_threshold_values"]])
        most_common_diff_u = diff_u_counter.most_common(5)
        
        if most_common_diff_u:
            values, counts = zip(*most_common_diff_u)
            bars7 = ax7.bar(range(len(values)), counts, color="lightpink")
            ax7.set_title("Most Frequent Optimal diff_u_threshold (Successful Tasks Only)")
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
        ax7.set_title("Optimal diff_u_threshold Values (Successful Tasks Only)")
    
    # Plot 8: Optimal parameter frequency for diff_v_threshold
    ax8 = plt.subplot(3, 4, 8)
    if statistics["optimal_diff_v_threshold_values"]:
        diff_v_counter = Counter([round(val, 8) for val in statistics["optimal_diff_v_threshold_values"]])
        most_common_diff_v = diff_v_counter.most_common(5)
        
        if most_common_diff_v:
            values, counts = zip(*most_common_diff_v)
            bars8 = ax8.bar(range(len(values)), counts, color="lightcyan")
            ax8.set_title("Most Frequent Optimal diff_v_threshold (Successful Tasks Only)")
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
        ax8.set_title("Optimal diff_v_threshold Values (Successful Tasks Only)")
    
    # Plot 9: Optimal parameter frequency for res_iter_v_threshold
    ax9 = plt.subplot(3, 4, 9)
    if statistics["optimal_res_iter_v_threshold_values"]:
        res_iter_counter = Counter([round(val, 8) for val in statistics["optimal_res_iter_v_threshold_values"]])
        most_common_res_iter = res_iter_counter.most_common(5)
        
        if most_common_res_iter:
            values, counts = zip(*most_common_res_iter)
            bars9 = ax9.bar(range(len(values)), counts, color="lightyellow")
            ax9.set_title("Most Frequent Optimal res_iter_v_threshold (Successful Tasks Only)")
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
        ax9.set_title("Optimal res_iter_v_threshold Values (Successful Tasks Only)")
    
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
    ax12.set_title("Parameter Distribution Summary (Successful Tasks Only)")
    ax12.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ns_channel_2d_statistics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed statistics file
    stats_file = os.path.join(output_dir, "ns_channel_2d_statistics_summary.txt")
    with open(stats_file, "w") as f:
        f.write("=== NS Channel 2D Dummy Search Statistics Summary ===\n\n")

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
            # Filter out infinite costs for average calculation
            finite_costs = [cost for cost in data["costs"] if not np.isinf(cost)]
            avg_cost = np.mean(finite_costs) if finite_costs else 0
            f.write(f"   {param}: {data['converged']}/{data['total']} ({rate:.2f}%), avg cost: {avg_cost:.0f}\n")
        f.write("\n")

        f.write("4. Convergence by Profile:\n")
        for profile, data in statistics["convergence_by_profile"].items():
            rate = (data["converged"] / data["total"] * 100) if data["total"] > 0 else 0
            f.write(f"   {profile}: {data['converged']}/{data['total']} ({rate:.2f}%)\n")
        f.write("\n")

        f.write("5. Optimal Parameter Frequencies (Successful Tasks Only):\n")
        if statistics["optimal_mesh_x_values"]:
            mesh_x_values, mesh_x_counts = np.unique(list(statistics["optimal_mesh_x_values"]), return_counts=True)
            f.write("   mesh_x parameter (iterative+0-shot):\n")
            for mesh_x, count in zip(mesh_x_values, mesh_x_counts):
                f.write(f"     mesh_x={mesh_x}: {count} times\n")

        if statistics["optimal_mesh_y_values"]:
            mesh_y_values, mesh_y_counts = np.unique(list(statistics["optimal_mesh_y_values"]), return_counts=True)
            f.write("   mesh_y parameter (iterative+0-shot):\n")
            for mesh_y, count in zip(mesh_y_values, mesh_y_counts):
                f.write(f"     mesh_y={mesh_y}: {count} times\n")

        if statistics["optimal_omega_u_values"]:
            omega_u_values, omega_u_counts = np.unique(list(statistics["optimal_omega_u_values"]), return_counts=True)
            f.write("   omega_u parameter (0-shot):\n")
            for omega_u, count in zip(omega_u_values, omega_u_counts):
                f.write(f"     omega_u={omega_u:.3f}: {count} times\n")

        if statistics["optimal_omega_v_values"]:
            omega_v_values, omega_v_counts = np.unique(list(statistics["optimal_omega_v_values"]), return_counts=True)
            f.write("   omega_v parameter (0-shot):\n")
            for omega_v, count in zip(omega_v_values, omega_v_counts):
                f.write(f"     omega_v={omega_v:.3f}: {count} times\n")

        if statistics["optimal_omega_p_values"]:
            omega_p_values, omega_p_counts = np.unique(list(statistics["optimal_omega_p_values"]), return_counts=True)
            f.write("   omega_p parameter (0-shot):\n")
            for omega_p, count in zip(omega_p_values, omega_p_counts):
                f.write(f"     omega_p={omega_p:.3f}: {count} times\n")

        if statistics["optimal_diff_u_threshold_values"]:
            diff_u_values, diff_u_counts = np.unique(list(statistics["optimal_diff_u_threshold_values"]), return_counts=True)
            f.write("   diff_u_threshold parameter (0-shot):\n")
            for diff_u, count in zip(diff_u_values, diff_u_counts):
                f.write(f"     diff_u_threshold={diff_u:.2e}: {count} times\n")

        if statistics["optimal_diff_v_threshold_values"]:
            diff_v_values, diff_v_counts = np.unique(list(statistics["optimal_diff_v_threshold_values"]), return_counts=True)
            f.write("   diff_v_threshold parameter (0-shot):\n")
            for diff_v, count in zip(diff_v_values, diff_v_counts):
                f.write(f"     diff_v_threshold={diff_v:.2e}: {count} times\n")

        if statistics["optimal_res_iter_v_threshold_values"]:
            res_iter_values, res_iter_counts = np.unique(list(statistics["optimal_res_iter_v_threshold_values"]), return_counts=True)
            f.write("   res_iter_v_threshold parameter (0-shot):\n")
            for res_iter, count in zip(res_iter_values, res_iter_counts):
                if isinstance(res_iter, str):
                    f.write(f"     res_iter_v_threshold={res_iter}: {count} times\n")
                else:
                    f.write(f"     res_iter_v_threshold={res_iter:.2e}: {count} times\n")


def process_mesh_task(target_param, target_config, task_params, profile, precision_name, precision_vals, statistics, successful_tasks, failed_tasks, mesh_base_values, wall_base_values):
    """Process a mesh task with proper aspect ratio handling."""
    # Call appropriate search function based on target parameter
    if target_param == "mesh_x":
        # For mesh_x, use initial_value and multiplication_factor for iterative search
        initial_value = target_config.get("initial_value", 256)
        multiplication_factor = target_config.get("multiplication_factor", 2)
        max_iteration_num = target_config.get("max_iteration_num", 6)
        
        # Generate mesh_x values using initial_value and multiplication_factor
        mesh_x_values = []
        current_value = initial_value
        for i in range(max_iteration_num):
            mesh_x_values.append(int(current_value))
            current_value *= multiplication_factor
        
        # For mesh_x task, use the mesh_y value that was already calculated in the main loop
        # to maintain the aspect ratio
        mesh_y = task_params["mesh_y"]
        mesh_y_values = []
        current_value = mesh_y
        
        for i in range(max_iteration_num):
            mesh_y_values.append(current_value)
            current_value *= multiplication_factor

        # Scale wall dimensions proportionally with mesh resolution
        other_params_list = None
        if "other_params" in task_params:
            other_params_list = []
            base_mesh_x = mesh_base_values["base_mesh_x"]
            base_mesh_y = mesh_base_values["base_mesh_y"]
            
            for i in range(max_iteration_num):
                current_mesh_x = mesh_x_values[i]
                current_mesh_y = mesh_y_values[i]
                
                # Calculate scaling factors based on current mesh vs base mesh
                scale_x = current_mesh_x / base_mesh_x
                scale_y = current_mesh_y / base_mesh_y
                
                scaled_wall_params = {
                    "wall_height": int(round(wall_base_values["base_wall_height"] * scale_y)),
                    "wall_width": int(round(wall_base_values["base_wall_width"] * scale_x)),
                    "wall_start_height": int(round(wall_base_values["base_wall_start_height"] * scale_y)),
                    "wall_start_width": int(round(wall_base_values["base_wall_start_width"] * scale_x))
                }
                other_params_list.append(scaled_wall_params)
            
        
        
        boundary_condition = get_boundary_condition(profile)
        is_converged, best_param, cost_history, param_history = grid_search_mesh_x(
            profile=profile,
            boundary_condition=boundary_condition,
            mesh_x_values=mesh_x_values,
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
            p_rmse_tolerance=precision_vals["p_rmse_tolerance"],
            other_params_list=other_params_list
        )
        if is_converged and best_param is not None:
            statistics["optimal_mesh_x_values"].append(best_param)

    elif target_param == "mesh_y":
        # For mesh_y, use initial_value and multiplication_factor for iterative search
        initial_value = target_config.get("initial_value", 64)
        multiplication_factor = target_config.get("multiplication_factor", 2)
        max_iteration_num = target_config.get("max_iteration_num", 6)
        
        # Generate mesh_y values using initial_value and multiplication_factor
        mesh_y_values = []
        current_value = initial_value
        for i in range(max_iteration_num):
            mesh_y_values.append(int(current_value))
            current_value *= multiplication_factor
        
        # For mesh_y task, use the mesh_x value that was already calculated in the main loop
        # to maintain the aspect ratio
        mesh_x = task_params["mesh_x"]
        mesh_x_values = []
        current_value = mesh_x
        for i in range(max_iteration_num):
            mesh_x_values.append(current_value)
            current_value *= multiplication_factor
        
        # Scale wall dimensions proportionally with mesh resolution
        other_params_list = None
        if "other_params" in task_params:
            other_params_list = []
            base_mesh_x = mesh_base_values["base_mesh_x"]
            base_mesh_y = mesh_base_values["base_mesh_y"]
            
            for i in range(max_iteration_num):
                current_mesh_x = mesh_x_values[i]
                current_mesh_y = mesh_y_values[i]
                
                # Calculate scaling factors based on current mesh vs base mesh
                scale_x = current_mesh_x / base_mesh_x
                scale_y = current_mesh_y / base_mesh_y
                
                scaled_wall_params = {
                    "wall_height": int(round(wall_base_values["base_wall_height"] * scale_y)),
                    "wall_width": int(round(wall_base_values["base_wall_width"] * scale_x)),
                    "wall_start_height": int(round(wall_base_values["base_wall_start_height"] * scale_y)),
                    "wall_start_width": int(round(wall_base_values["base_wall_start_width"] * scale_x))
                }
                other_params_list.append(scaled_wall_params)
            
        boundary_condition = get_boundary_condition(profile)
        is_converged, best_param, cost_history, param_history = grid_search_mesh_y(
            profile=profile,
            boundary_condition=boundary_condition,
            mesh_x_values=mesh_x_values,
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
            p_rmse_tolerance=precision_vals["p_rmse_tolerance"],
            other_params_list=other_params_list
        )
        if is_converged and best_param is not None:
            statistics["optimal_mesh_y_values"].append(best_param)

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
            "multiplication_factor": target_config.get("multiplication_factor"),
            "max_iteration_num": target_config.get("max_iteration_num"),
            "search_range": target_config.get("search_range"),
            "search_range_slice_num": target_config.get("search_range_slice_num"),
            "schedule_options": target_config.get("schedule_options")
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


def process_non_mesh_task(target_param, target_config, task_params, profile, precision_name, precision_vals, statistics, successful_tasks, failed_tasks):
    """Process a non-mesh task with the original logic."""
    # Call appropriate search function based on target parameter
    if target_param == "omega_u":
        # For omega_u, use search_range for 0-shot search
        search_range = target_config.get("search_range", [0.1, 1.0])
        search_range_min = search_range[0]
        search_range_max = search_range[1]
        search_range_slice_num = target_config.get("search_range_slice_num", 10)
        
        # Generate omega_u values using search_range
        omega_u_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
        
        boundary_condition = get_boundary_condition(profile)
        print(f"        Using mesh: mesh_x={task_params['mesh_x']}, mesh_y={task_params['mesh_y']}")
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
            p_rmse_tolerance=precision_vals["p_rmse_tolerance"],
            other_params=task_params.get("other_params")
        )
        if is_converged and best_param is not None:
            statistics["optimal_omega_u_values"].append(best_param)

    elif target_param == "omega_v":
        # For omega_v, use search_range for 0-shot search
        search_range = target_config.get("search_range", [0.1, 1.0])
        search_range_min = search_range[0]
        search_range_max = search_range[1]
        search_range_slice_num = target_config.get("search_range_slice_num", 10)
        
        # Generate omega_v values using search_range
        omega_v_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
        
        boundary_condition = get_boundary_condition(profile)
        print(f"        Using mesh: mesh_x={task_params['mesh_x']}, mesh_y={task_params['mesh_y']}")
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
            p_rmse_tolerance=precision_vals["p_rmse_tolerance"],
            other_params=task_params.get("other_params")
        )
        if is_converged and best_param is not None:
            statistics["optimal_omega_v_values"].append(best_param)

    elif target_param == "omega_p":
        # For omega_p, use search_range for 0-shot search
        search_range = target_config.get("search_range", [0.1, 0.5])
        search_range_min = search_range[0]
        search_range_max = search_range[1]
        search_range_slice_num = target_config.get("search_range_slice_num", 8)
        
        # Generate omega_p values using search_range
        omega_p_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
        
        boundary_condition = get_boundary_condition(profile)
        print(f"        Using mesh: mesh_x={task_params['mesh_x']}, mesh_y={task_params['mesh_y']}")
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
            p_rmse_tolerance=precision_vals["p_rmse_tolerance"],
            other_params=task_params.get("other_params")
        )
        if is_converged and best_param is not None:
            statistics["optimal_omega_p_values"].append(best_param)

    elif target_param == "diff_u_threshold":
        # For diff_u_threshold, use search_range for 0-shot search
        search_range = target_config.get("search_range", [1e-07, 1e-03])
        search_range_min = search_range[0]
        search_range_max = search_range[1]
        search_range_slice_num = target_config.get("search_range_slice_num", 5)
        
        # Generate diff_u_threshold values using search_range (logarithmic spacing)
        # Reverse order to start with looser thresholds and work towards tighter ones
        diff_u_values = np.logspace(np.log10(float(search_range_min)), np.log10(float(search_range_max)), search_range_slice_num)[::-1]
        
        boundary_condition = get_boundary_condition(profile)
        print(f"        Using mesh: mesh_x={task_params['mesh_x']}, mesh_y={task_params['mesh_y']}")
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
            p_rmse_tolerance=precision_vals["p_rmse_tolerance"],
            other_params=task_params.get("other_params")
        )
        if is_converged and best_param is not None:
            statistics["optimal_diff_u_threshold_values"].append(best_param)

    elif target_param == "diff_v_threshold":
        # For diff_v_threshold, use search_range for 0-shot search
        search_range = target_config.get("search_range", [1e-07, 1e-03])
        search_range_min = search_range[0]
        search_range_max = search_range[1]
        search_range_slice_num = target_config.get("search_range_slice_num", 5)
        
        # Generate diff_v_threshold values using search_range (logarithmic spacing)
        # Reverse order to start with looser thresholds and work towards tighter ones
        diff_v_values = np.logspace(np.log10(float(search_range_min)), np.log10(float(search_range_max)), search_range_slice_num)[::-1]
        
        boundary_condition = get_boundary_condition(profile)
        print(f"        Using mesh: mesh_x={task_params['mesh_x']}, mesh_y={task_params['mesh_y']}")
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
            p_rmse_tolerance=precision_vals["p_rmse_tolerance"],
            other_params=task_params.get("other_params")
        )
        if is_converged and best_param is not None:
            statistics["optimal_diff_v_threshold_values"].append(best_param)

    elif target_param == "res_iter_v_threshold":
        # For res_iter_v_threshold, use search_range for 0-shot search
        search_range = target_config.get("search_range", [1e-07, 1e-03])
        search_range_min = search_range[0]
        search_range_max = search_range[1]
        search_range_slice_num = target_config.get("search_range_slice_num", 5)
        
        # Generate res_iter_v_threshold values using search_range (logarithmic spacing)
        # Reverse order to start with looser thresholds and work towards tighter ones
        res_iter_v_values = np.logspace(np.log10(float(search_range_min)), np.log10(float(search_range_max)), search_range_slice_num)[::-1]
        
        # Add schedule options if available
        schedule_options = target_config.get("schedule_options", ["exp_decay"])
        if schedule_options:
            res_iter_v_values = list(res_iter_v_values) + schedule_options
        
        boundary_condition = get_boundary_condition(profile)
        print(f"        Using mesh: mesh_x={task_params['mesh_x']}, mesh_y={task_params['mesh_y']}")
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
            p_rmse_tolerance=precision_vals["p_rmse_tolerance"],
            other_params=task_params.get("other_params")
        )
        if is_converged and best_param is not None:
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
            "multiplication_factor": target_config.get("multiplication_factor"),
            "max_iteration_num": target_config.get("max_iteration_num"),
            "search_range": target_config.get("search_range"),
            "search_range_slice_num": target_config.get("search_range_slice_num"),
            "schedule_options": target_config.get("schedule_options")
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


def calculate_scaled_wall_dimensions(mesh_x, mesh_y, wall_base_values, mesh_base_values):
    """Calculate wall dimensions scaled proportionally with mesh resolution"""
    # Calculate scaling factors
    scale_x = mesh_x / mesh_base_values["base_mesh_x"]
    scale_y = mesh_y / mesh_base_values["base_mesh_y"]
    
    # Scale wall dimensions proportionally
    scaled_wall_height = int(round(wall_base_values["base_wall_height"] * scale_y))
    scaled_wall_width = int(round(wall_base_values["base_wall_width"] * scale_x))
    scaled_wall_start_height = int(round(wall_base_values["base_wall_start_height"] * scale_y))
    scaled_wall_start_width = int(round(wall_base_values["base_wall_start_width"] * scale_x))
    
    return {
        "wall_height": scaled_wall_height,
        "wall_width": scaled_wall_width,
        "wall_start_height": scaled_wall_start_height,
        "wall_start_width": scaled_wall_start_width
    }


def process_task_at_all_precisions(target_param, target_config, task_params, profile, precision_levels, statistics, successful_tasks, failed_tasks, wall_base_values, mesh_base_values):
    """Process a task at all precision levels"""
    # Process from high to low precision
    precision_order = ["high", "medium", "low"]
    
    for precision_name in precision_order:
        if precision_name not in precision_levels:
            continue
            
        precision_vals = precision_levels[precision_name]
        
        # Process the task
        if target_param in ["mesh_x", "mesh_y"]:
            # For mesh tasks, we don't have mesh_x/mesh_y in task_params yet, so we'll handle wall scaling in the mesh task functions
            process_mesh_task(target_param, target_config, task_params, profile, precision_name, precision_vals, statistics, successful_tasks, failed_tasks, mesh_base_values, wall_base_values)
        else:
            # For non-mesh tasks, calculate scaled wall dimensions for this mesh configuration
            mesh_x = task_params["mesh_x"]
            mesh_y = task_params["mesh_y"]
            scaled_wall_params = calculate_scaled_wall_dimensions(mesh_x, mesh_y, wall_base_values, mesh_base_values)
            
            # Add scaled wall parameters to task_params
            task_params_with_walls = task_params.copy()
            task_params_with_walls["other_params"] = scaled_wall_params
            
            # Calculate and print ratios
            wall_height = scaled_wall_params["wall_height"]
            wall_width = scaled_wall_params["wall_width"]
            wall_height_ratio = wall_height / mesh_y
            wall_width_ratio = wall_width / mesh_x
            
            print(f"        Scaled wall dimensions: {scaled_wall_params}")
            print(f"        Wall ratios: wall_height/mesh_y={wall_height_ratio:.3f}, wall_width/mesh_x={wall_width_ratio:.3f}")
            
            process_non_mesh_task(target_param, target_config, task_params_with_walls, profile, precision_name, precision_vals, statistics, successful_tasks, failed_tasks)


def main():
    print("=== NS Channel 2D Dummy Solution Generation ===")
    print("Loading configuration from ns_channel_2d.yaml...")

    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "ns_channel_2d.yaml")
    config = load_config(config_path)
    print("✅ Configuration loaded successfully")

    # Extract configuration sections
    target_configs = build_target_configs(config)
    precision_configs = config["precision_levels"]
    profiles = config["profiles"]["active_profiles"]
    
    # Load non-target combinations from configuration
    non_target_config = config.get("non_target_combinations", {})
    aspect_ratios = non_target_config.get("aspect_ratios", [1.0])
    mesh_combinations = non_target_config.get("mesh_combinations", [[64, 16], [128, 32], [192, 48], [256, 64]])
    mesh_base_values = non_target_config.get("mesh_base_values", {"base_mesh_x": 64, "base_mesh_y": 32})
    wall_base_values = non_target_config.get("wall_base_values", {"base_wall_height": 10, "base_wall_width": 10, "base_wall_start_height": 20, "base_wall_start_width": 80})
    
    print(f"📊 Active precision levels: {list(precision_configs.keys())}")
    print(f"📁 Active profiles: {profiles}")
    print(f"🎯 Target parameters: {list(target_configs.keys())}")
    print(f"📐 Aspect ratios for mesh tasks: {aspect_ratios}")
    print(f"🔲 Mesh combinations: {mesh_combinations}")
    print(f"📏 Base mesh values: {mesh_base_values}")
    print(f"🧱 Base wall values: {wall_base_values}")
    print("Generating all cached results for LLM automation tasks...")

    # Change to repository root directory
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    print(f"Working directory: {os.getcwd()}")
    
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
    for profile in profiles:
        print(f"\n--- Processing Profile: {profile} ---")

        for target_param, target_config in target_configs.items():
            print(f"  Target parameter: {target_param}")

            # Get all non-target parameter names and their value lists
            non_target_params = target_config["non_target_parameters"]
            param_names = list(non_target_params.keys())
            param_values = [non_target_params[name] for name in param_names]

            # Generate all combinations using nested loops (cartesian product)
            for combination in itertools.product(*param_values):
                # Build parameters dictionary
                task_params = dict(zip(param_names, combination))

                # For mesh tasks, we need to handle aspect ratios
                if target_param in ["mesh_x", "mesh_y"]:
                    for aspect_ratio in aspect_ratios:
                        aspect_task_params = task_params.copy()
                        
                        if target_param == "mesh_x":
                            # For mesh_x task, calculate mesh_y based on aspect ratio
                            if aspect_ratio > 0.5:
                                continue
                            base_mesh_x = mesh_base_values["base_mesh_x"]
                            aspect_task_params["mesh_y"] = int(base_mesh_x * aspect_ratio)
                            scaled_wall_params = calculate_scaled_wall_dimensions(base_mesh_x, aspect_task_params["mesh_y"], wall_base_values, mesh_base_values)
                            aspect_task_params["other_params"] = scaled_wall_params
                        else:  # mesh_y task
                            # For mesh_y task, calculate mesh_x based on aspect ratio
                            if aspect_ratio < 0.2:
                                continue
                            base_mesh_y = mesh_base_values["base_mesh_y"]
                            aspect_task_params["mesh_x"] = int(base_mesh_y / aspect_ratio)
                            scaled_wall_params = calculate_scaled_wall_dimensions(aspect_task_params["mesh_x"], base_mesh_y, wall_base_values, mesh_base_values)
                            aspect_task_params["other_params"] = scaled_wall_params
                        
                        # Add aspect_ratio to task parameters
                        aspect_task_params["aspect_ratio"] = aspect_ratio
                        
                        # Process this task at all precision levels
                        process_task_at_all_precisions(target_param, target_config, aspect_task_params, profile, 
                                                      precision_configs, statistics, successful_tasks, failed_tasks,
                                                      wall_base_values, mesh_base_values)
                else:
                    # For non-mesh tasks, convert mesh_combination index to actual mesh_x and mesh_y pairs
                    mesh_combination_idx = task_params["mesh_combination"]
                    mesh_x, mesh_y = mesh_combinations[mesh_combination_idx]
                    
                    # Update task_params with actual mesh values
                    task_params["mesh_x"] = mesh_x
                    task_params["mesh_y"] = mesh_y
                    
                    # Process this task at all precision levels
                    process_task_at_all_precisions(target_param, target_config, task_params, profile, 
                                                  precision_configs, statistics, successful_tasks, failed_tasks,
                                                  wall_base_values, mesh_base_values)

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
        
        # For mesh tasks, multiply by aspect ratios
        if target_param in ["mesh_x", "mesh_y"]:
            combinations_per_target *= (len(aspect_ratios) - 1)
        
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
        
        # For mesh tasks, multiply by aspect ratios
        if target_param in ["mesh_x", "mesh_y"]:
            combinations_per_target *= len(aspect_ratios)
        
        tasks_for_param = len(profiles) * combinations_per_target
        print(f"  {target_param}: {len(profiles)} profiles × {combinations_per_target} combos = {tasks_for_param}")
    print(f"  Expected total per precision: {expected_total}")
    print(f"  Expected total across {len(precision_configs)} precisions: {expected_total * len(precision_configs)}")
    print(f"  Actual total: {statistics['total_tasks']}")
    
    print(f"\n{'='*60}")
    print("NS CHANNEL 2D DUMMY SOLUTION GENERATION COMPLETED!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
