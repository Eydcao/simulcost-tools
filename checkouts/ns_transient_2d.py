#!/usr/bin/env python3
"""
NS Transient 2D Dummy Solution Generation Script

This script generates dummy solutions for all parameter combinations defined in the
ns_transient_2d.yaml file. It uses direct Python function calls instead of subprocess
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
from dummy_sols.ns_transient_2d import (
    grid_search_resolution, grid_search_cfl, grid_search_relaxation_factor,
    grid_search_residual_threshold
)


def get_profile_environment_params(profile):
    """Get all environment parameters from profile config file"""
    config_path = f"run_configs/ns_transient_2d/{profile}.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Extract environment dependent parameters
            env_params = {
                'boundary_condition': config.get('boundary_condition'),
                'reynolds_num': config.get('reynolds_num'),
                'vorticity_confinement': config.get('vorticity_confinement'),
                'total_runtime': config.get('total_runtime'),
                'no_dye': config.get('no_dye'),
                'cpu': config.get('cpu'),
                'visualization': config.get('visualization'),
                'advection_scheme': config.get('advection_scheme')
            }
            
            # Check if all required parameters are present
            missing_params = [k for k, v in env_params.items() if v is None]
            if missing_params:
                raise ValueError(f"Missing parameters in config file {config_path}: {missing_params}")
                
            return env_params
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_path} not found. Please ensure the profile '{profile}' exists.")


def save_datasets(successful_tasks, failed_tasks, output_dir):
    """Save successful and failed tasks as separate JSON datasets in subfolders"""
    # Create subfolders for successful and failed tasks
    success_dir = os.path.join(output_dir, "ns_transient_2d", "successful")
    failed_dir = os.path.join(output_dir, "ns_transient_2d", "failed")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    
    # Save successful tasks (overwrite existing file)
    success_file = os.path.join(success_dir, "tasks.json")
    with open(success_file, "w") as f:  # "w" mode overwrites existing file
        json.dump({
            "metadata": {
                "solver": "ns_transient_2d",
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
                "solver": "ns_transient_2d", 
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
    
    # Plot 5: Optimal parameter frequency for resolution
    ax5 = plt.subplot(3, 4, 5)
    if statistics["optimal_resolution_values"]:
        resolution_counter = Counter([int(val) for val in statistics["optimal_resolution_values"]])
        most_common_resolution = resolution_counter.most_common(5)
        if most_common_resolution:
            values, counts = zip(*most_common_resolution)
            bars5 = ax5.bar(range(len(values)), counts, color="lightsteelblue")
            ax5.set_title("Most Frequent Optimal Resolution")
            ax5.set_xlabel("Resolution Value")
            ax5.set_ylabel("Frequency")
            ax5.set_xticks(range(len(values)))
            ax5.set_xticklabels([str(v) for v in values], rotation=45, ha='right')
            
            for bar, count in zip(bars5, counts):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                        str(count), ha='center', va='bottom')
    else:
        ax5.text(0.5, 0.5, 'No optimal resolution\nvalues found', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title("Optimal Resolution Values")
    
    # Plot 6: Optimal parameter frequency for cfl
    ax6 = plt.subplot(3, 4, 6)
    if statistics["optimal_cfl_values"]:
        cfl_counter = Counter([round(val, 3) for val in statistics["optimal_cfl_values"]])
        most_common_cfl = cfl_counter.most_common(5)
        if most_common_cfl:
            values, counts = zip(*most_common_cfl)
            bars6 = ax6.bar(range(len(values)), counts, color="lightcoral")
            ax6.set_title("Most Frequent Optimal CFL")
            ax6.set_xlabel("CFL Value")
            ax6.set_ylabel("Frequency")
            ax6.set_xticks(range(len(values)))
            ax6.set_xticklabels([f"{v:.3f}" for v in values], rotation=45, ha='right')
            
            for bar, count in zip(bars6, counts):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                        str(count), ha='center', va='bottom')
    else:
        ax6.text(0.5, 0.5, 'No optimal CFL\nvalues found', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title("Optimal CFL Values")
    
    # Plot 7: Optimal parameter frequency for relaxation_factor
    ax7 = plt.subplot(3, 4, 7)
    if statistics["optimal_relaxation_factor_values"]:
        relaxation_counter = Counter([round(val, 2) for val in statistics["optimal_relaxation_factor_values"]])
        most_common_relaxation = relaxation_counter.most_common(5)
        if most_common_relaxation:
            values, counts = zip(*most_common_relaxation)
            bars7 = ax7.bar(range(len(values)), counts, color="lightpink")
            ax7.set_title("Most Frequent Optimal Relaxation Factor")
            ax7.set_xlabel("Relaxation Factor")
            ax7.set_ylabel("Frequency")
            ax7.set_xticks(range(len(values)))
            ax7.set_xticklabels([f"{v:.2f}" for v in values], rotation=45, ha='right')
            
            for bar, count in zip(bars7, counts):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                        str(count), ha='center', va='bottom')
    else:
        ax7.text(0.5, 0.5, 'No optimal relaxation\nfactor values found', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title("Optimal Relaxation Factor Values")
    
    # Plot 8: Optimal parameter frequency for residual_threshold
    ax8 = plt.subplot(3, 4, 8)
    if statistics["optimal_residual_threshold_values"]:
        residual_counter = Counter([round(val, 8) for val in statistics["optimal_residual_threshold_values"]])
        most_common_residual = residual_counter.most_common(5)
        if most_common_residual:
            values, counts = zip(*most_common_residual)
            bars8 = ax8.bar(range(len(values)), counts, color="lightcyan")
            ax8.set_title("Most Frequent Optimal Residual Threshold")
            ax8.set_xlabel("Residual Threshold")
            ax8.set_ylabel("Frequency")
            ax8.set_xticks(range(len(values)))
            ax8.set_xticklabels([f"{v:.1e}" for v in values], rotation=45, ha='right')
            
            for bar, count in zip(bars8, counts):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                        str(count), ha='center', va='bottom')
    else:
        ax8.text(0.5, 0.5, 'No optimal residual\nthreshold values found', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=12)
        ax8.set_title("Optimal Residual Threshold Values")
    
    # Plot 9: Empty plot for future use
    ax9 = plt.subplot(3, 4, 9)
    ax9.text(0.5, 0.5, 'Reserved for\nfuture analysis', 
            ha='center', va='center', transform=ax9.transAxes, fontsize=12)
    ax9.set_title("Future Analysis")
    ax9.axis('off')
    
    # Plot 10: Parameter convergence comparison
    ax10 = plt.subplot(3, 4, 10)
    param_names = list(statistics["convergence_by_target"].keys())
    convergence_counts = []
    for param in param_names:
        converged = statistics["convergence_by_target"][param]["converged"]
        total = statistics["convergence_by_target"][param]["total"]
        convergence_counts.append(converged)
    
    bars10 = ax10.bar(param_names, convergence_counts, color=["skyblue", "lightgreen", "lightcoral", "gold"])
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
    
    bars11 = ax11.bar(param_names, avg_costs, color=["skyblue", "lightgreen", "lightcoral", "gold"])
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
    for param_name, values in [("resolution", statistics["optimal_resolution_values"]), 
                               ("cfl", statistics["optimal_cfl_values"]),
                               ("relaxation_factor", statistics["optimal_relaxation_factor_values"]),
                               ("residual_threshold", statistics["optimal_residual_threshold_values"])]:
        if values:
            if param_name in ["resolution"]:
                counter = Counter([int(val) for val in values])
            elif param_name in ["cfl", "relaxation_factor"]:
                counter = Counter([round(val, 2) for val in values])
            elif param_name in ["residual_threshold"]:
                counter = Counter([round(val, 6) for val in values])
                
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
    plt.savefig(os.path.join(output_dir, "ns_transient_2d_statistics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate summary text file
    with open(os.path.join(output_dir, "ns_transient_2d_statistics_summary.txt"), "w") as f:
        f.write("NS TRANSIENT 2D DUMMY SOLUTION STATISTICS SUMMARY\n")
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
        for param_name, values in [("resolution", statistics["optimal_resolution_values"]), 
                                   ("cfl", statistics["optimal_cfl_values"]),
                                   ("relaxation_factor", statistics["optimal_relaxation_factor_values"]),
                                   ("residual_threshold", statistics["optimal_residual_threshold_values"])]:
            if values:
                if param_name in ["resolution"]:
                    counter = Counter([int(val) for val in values])
                elif param_name in ["cfl", "relaxation_factor"]:
                    counter = Counter([round(val, 2) for val in values])
                elif param_name in ["residual_threshold"]:
                    counter = Counter([round(val, 6) for val in values])
                    
                most_common = counter.most_common(1)[0]
                frequency = most_common[1] / len(values) * 100
                f.write(f"  {param_name}: {most_common[0]} (appears {frequency:.1f}% of the time)\n")
            else:
                f.write(f"  {param_name}: No values found\n")


def process_task(target_param, target_config, task_params, profile, precision_name, precision_vals, statistics, successful_tasks, failed_tasks):
    """Process a single task with the appropriate grid search function."""
    # Load environment parameters from profile config
    env_params = get_profile_environment_params(profile)
    
    # Call appropriate search function based on target parameter
    if target_param == "resolution":
        # For resolution, use initial_value and multiplication_factor for iterative search
        initial_value = target_config.get("initial_value", 200)
        multiplication_factor = target_config.get("multiplication_factor", 2)
        max_iteration_num = target_config.get("max_iteration_num", 4)
        
        # Generate resolution values using initial_value and multiplication_factor
        resolution_values = []
        current_value = initial_value
        for i in range(max_iteration_num):
            resolution_values.append(int(current_value))
            current_value *= multiplication_factor
        
        is_converged, best_param, cost_history, param_history = grid_search_resolution(
            profile=profile,
            boundary_condition=env_params["boundary_condition"],
            resolution_values=resolution_values,
            reynolds_num=env_params["reynolds_num"],
            cfl=task_params["cfl"],
            relaxation_factor=task_params["relaxation_factor"],
            residual_threshold=task_params["residual_threshold"],
            total_runtime=env_params["total_runtime"],
            norm_rmse_tolerance=precision_vals["norm_rmse_tolerance"],
            other_params=task_params.get("other_params")
        )
        if best_param is not None:
            statistics["optimal_resolution_values"].append(best_param)

    elif target_param == "cfl":
        # For cfl, use initial_value and multiplication_factor for iterative search
        initial_value = target_config.get("initial_value", 0.8)
        multiplication_factor = target_config.get("multiplication_factor", 0.5)
        max_iteration_num = target_config.get("max_iteration_num", 5)
        
        # Generate cfl values from YAML configuration
        if "exact_values" in target_config:
            cfl_values = target_config["exact_values"]
            print(f"Using exact_values from YAML for CFL: {cfl_values}")
        else:
            # Fallback to iterative generation if exact_values not specified
            cfl_values = []
            current_value = initial_value
            for i in range(max_iteration_num):
                cfl_values.append(round(current_value, 3))  # Round to avoid floating-point precision issues
                current_value *= multiplication_factor
                # Stop if we reach minimum value of 0.05
                if current_value < 0.05:
                    break
        
        is_converged, best_param, cost_history, param_history = grid_search_cfl(
            profile=profile,
            boundary_condition=env_params["boundary_condition"],
            resolution=task_params["resolution"],
            reynolds_num=env_params["reynolds_num"],
            cfl_values=cfl_values,
            relaxation_factor=task_params["relaxation_factor"],
            residual_threshold=task_params["residual_threshold"],
            total_runtime=env_params["total_runtime"],
            norm_rmse_tolerance=precision_vals["norm_rmse_tolerance"],
            other_params=task_params.get("other_params")
        )
        if best_param is not None:
            statistics["optimal_cfl_values"].append(best_param)

    elif target_param == "relaxation_factor":
        # For relaxation_factor, use search_range for 0-shot search
        search_range = target_config.get("search_range", [0.5, 2.0])
        search_range_min = search_range[0]
        search_range_max = search_range[1]
        search_range_slice_num = target_config.get("search_range_slice_num", 8)
        
        # Generate relaxation_factor values from YAML configuration
        if "exact_values" in target_config:
            relaxation_factor_values = np.array(target_config["exact_values"])
            print(f"Using exact_values from YAML for relaxation_factor: {relaxation_factor_values}")
        else:
            # Fallback to linspace if exact_values not specified
            relaxation_factor_values = np.linspace(search_range_min, search_range_max, search_range_slice_num)
            # Round to avoid floating-point precision issues (e.g., 1.2000000002 -> 1.2)
            relaxation_factor_values = np.round(relaxation_factor_values, 2)
        
        is_converged, best_param, cost_history, param_history = grid_search_relaxation_factor(
            profile=profile,
            boundary_condition=env_params["boundary_condition"],
            resolution=task_params["resolution"],
            reynolds_num=env_params["reynolds_num"],
            cfl=task_params["cfl"],
            relaxation_factor_values=relaxation_factor_values,
            residual_threshold=task_params["residual_threshold"],
            total_runtime=env_params["total_runtime"],
            norm_rmse_tolerance=precision_vals["norm_rmse_tolerance"],
            other_params=task_params.get("other_params")
        )
        if best_param is not None:
            statistics["optimal_relaxation_factor_values"].append(best_param)

    elif target_param == "residual_threshold":
        # For residual_threshold, use search_range for 0-shot search
        search_range = target_config.get("search_range", [1e-01, 1e-05])
        search_range_min = search_range[0]
        search_range_max = search_range[1]
        search_range_slice_num = target_config.get("search_range_slice_num", 5)
        
        # Generate residual_threshold values from YAML configuration
        if "exact_values" in target_config:
            residual_threshold_values = np.array(target_config["exact_values"])
            print(f"Using exact_values from YAML: {residual_threshold_values}")
        else:
            # Fallback to logspace if exact_values not specified
            residual_threshold_values = np.logspace(np.log10(float(search_range_min)), np.log10(float(search_range_max)), search_range_slice_num)
            residual_threshold_values = np.round(residual_threshold_values, 6)
            print(f"Using logspace fallback: {residual_threshold_values}")
        
        is_converged, best_param, cost_history, param_history = grid_search_residual_threshold(
            profile=profile,
            boundary_condition=env_params["boundary_condition"],
            resolution=task_params["resolution"],
            reynolds_num=env_params["reynolds_num"],
            cfl=task_params["cfl"],
            relaxation_factor=task_params["relaxation_factor"],
            residual_threshold_values=residual_threshold_values,
            total_runtime=env_params["total_runtime"],
            norm_rmse_tolerance=precision_vals["norm_rmse_tolerance"],
            other_params=task_params.get("other_params")
        )
        if best_param is not None:
            statistics["optimal_residual_threshold_values"].append(best_param)

    # Create task record for dataset
    task_record = {
        "solver": "ns_transient_2d",
        "target_parameter": target_param,
        "profile": profile,
        "precision_level": precision_name,
        "precision_config": {
            "norm_rmse_tolerance": precision_vals["norm_rmse_tolerance"]
        },
        "target_config": {
            "initial_value": target_config.get("initial_value"),
            "multiplication_factor": target_config.get("multiplication_factor"),
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


def main():
    """Main execution function."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("NS TRANSIENT 2D DUMMY SOLUTION GENERATION")
    print("=" * 60)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "ns_transient_2d.yaml")
    config = load_config(config_path)
    target_configs = build_target_configs(config)
    precision_configs = config["precision_levels"]
    profiles = config["profiles"]["active_profiles"]
    
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
        "optimal_resolution_values": [],
        "optimal_cfl_values": [],
        "optimal_relaxation_factor_values": [],
        "optimal_residual_threshold_values": []
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

                # Process this task at all precision levels
                for precision_name, precision_vals in precision_configs.items():
                    print(f"    Precision: {precision_name}")
                    print(f"      Running {target_param} search with params: {task_params}")
                    
                    process_task(target_param, target_config, task_params, profile, precision_name, precision_vals, statistics, successful_tasks, failed_tasks)

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
    print("NS TRANSIENT 2D DUMMY SOLUTION GENERATION COMPLETED!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
