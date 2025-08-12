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
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checkouts.config_utils import load_config, build_target_configs
from dummy_sols.burgers_1d import find_convergent_cfl, find_convergent_n_space, find_optimal_k, find_optimal_w


def run_single_task(target_param, param_config, profile, precision_level, precision_config):
    """Run a single parameter optimization task."""
    print(f"\n{'='*60}")
    print(f"Running {target_param} optimization for profile {profile} at {precision_level} precision")
    print(f"{'='*60}")

    # Extract tolerances
    tolerance_linf = precision_config["tolerance_linf"]
    tolerance_rmse = precision_config["tolerance_rmse"]

    start_time = time.time()

    try:
        if target_param == "cfl":
            # CFL convergence search - iterative+0-shot
            non_target = param_config["non_target_parameters"]
            k_values = non_target["k"] if isinstance(non_target["k"], list) else [non_target["k"]]
            w_values = non_target["w"] if isinstance(non_target["w"], list) else [non_target["w"]]

            results = []
            for k in k_values:
                for w in w_values:
                    print(f"\nTesting CFL convergence with k={k}, w={w}")
                    is_converged, best_cfl, cost_history, param_history = find_convergent_cfl(
                        profile=profile,
                        cfl=param_config["initial_value"],
                        k=k,
                        w=w,
                        tolerance_infity=tolerance_linf,
                        tolerance_2=tolerance_rmse,
                    )

                    results.append(
                        {
                            "converged": is_converged,
                            "best_param": best_cfl,
                            "total_cost": sum(cost_history) if cost_history else 0,
                            "k": k,
                            "w": w,
                        }
                    )

            return results

        elif target_param == "n_space":
            # n_space convergence search - iterative+0-shot
            non_target = param_config["non_target_parameters"]
            k_values = non_target["k"] if isinstance(non_target["k"], list) else [non_target["k"]]
            w_values = non_target["w"] if isinstance(non_target["w"], list) else [non_target["w"]]
            cfl = non_target["cfl"]

            results = []
            for k in k_values:
                for w in w_values:
                    print(f"\nTesting n_space convergence with cfl={cfl}, k={k}, w={w}")
                    is_converged, best_n_space, cost_history, param_history = find_convergent_n_space(
                        profile=profile,
                        n_space=param_config["initial_value"],
                        cfl=cfl,
                        k=k,
                        w=w,
                        tolerance_infity=tolerance_linf,
                        tolerance_2=tolerance_rmse,
                    )

                    results.append(
                        {
                            "converged": is_converged,
                            "best_param": best_n_space,
                            "total_cost": sum(cost_history) if cost_history else 0,
                            "cfl": cfl,
                            "k": k,
                            "w": w,
                        }
                    )

            return results

        elif target_param == "k":
            # k parameter optimization - 0-shot with CFL search
            non_target = param_config["non_target_parameters"]
            w_values = non_target["w"] if isinstance(non_target["w"], list) else [non_target["w"]]

            results = []
            for w in w_values:
                print(f"\nTesting k parameter optimization with w={w}")
                is_converged, optimal_params, optimal_cost_history, param_history = find_optimal_k(
                    profile=profile, w=w, tolerance_infity=tolerance_linf, tolerance_2=tolerance_rmse
                )

                optimal_k, optimal_cfl = optimal_params if optimal_params != (None, None) else (None, None)

                results.append(
                    {
                        "converged": is_converged,
                        "best_param": optimal_k,
                        "optimal_cfl": optimal_cfl,
                        "total_cost": sum(optimal_cost_history) if optimal_cost_history else 0,
                        "w": w,
                    }
                )

            return results

        elif target_param == "w":
            # w parameter optimization - 0-shot with CFL search
            non_target = param_config["non_target_parameters"]
            k_values = non_target["k"] if isinstance(non_target["k"], list) else [non_target["k"]]

            results = []
            for k in k_values:
                print(f"\nTesting w parameter optimization with k={k}")
                is_converged, optimal_params, optimal_cost_history, param_history = find_optimal_w(
                    profile=profile, k=k, tolerance_infity=tolerance_linf, tolerance_2=tolerance_rmse
                )

                optimal_w, optimal_cfl = optimal_params if optimal_params != (None, None) else (None, None)

                results.append(
                    {
                        "converged": is_converged,
                        "best_param": optimal_w,
                        "optimal_cfl": optimal_cfl,
                        "total_cost": sum(optimal_cost_history) if optimal_cost_history else 0,
                        "k": k,
                    }
                )

            return results

    except Exception as e:
        print(f"Error in {target_param} optimization: {str(e)}")
        return [{"converged": False, "best_param": None, "total_cost": 0, "error": str(e)}]

    finally:
        elapsed_time = time.time() - start_time
        print(f"Task completed in {elapsed_time:.2f} seconds")


def plot_statistics(all_results):
    """Generate comprehensive statistical plots."""
    print("\nGenerating statistical plots...")

    # Prepare data for plotting
    param_convergence_rates = {}
    param_costs = defaultdict(list)
    optimal_param_counts = defaultdict(Counter)

    for result in all_results:
        target_param = result["target_param"]

        if target_param not in param_convergence_rates:
            param_convergence_rates[target_param] = {"converged": 0, "total": 0}

        for task_result in result["results"]:
            param_convergence_rates[target_param]["total"] += 1

            if task_result.get("converged", False):
                param_convergence_rates[target_param]["converged"] += 1

                # Track costs for converged cases
                param_costs[target_param].append(task_result.get("total_cost", 0))

                # Track optimal parameter frequencies
                best_param = task_result.get("best_param")
                if best_param is not None:
                    # Round to reasonable precision for counting
                    if target_param in ["k", "w"]:
                        best_param = round(best_param, 2)
                    elif target_param == "cfl":
                        best_param = round(best_param, 4)
                    elif target_param == "n_space":
                        best_param = int(best_param)

                    optimal_param_counts[target_param][best_param] += 1

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Burgers 1D Dummy Solution Statistics", fontsize=16, fontweight="bold")

    # Plot 1: Convergence rates
    params = list(param_convergence_rates.keys())
    convergence_rates = [
        param_convergence_rates[p]["converged"] / max(param_convergence_rates[p]["total"], 1) for p in params
    ]

    bars1 = ax1.bar(params, convergence_rates, color=["skyblue", "lightgreen", "lightcoral", "gold"][: len(params)])
    ax1.set_title("Convergence Rate by Parameter")
    ax1.set_ylabel("Convergence Rate")
    ax1.set_ylim(0, 1)

    # Add percentage labels on bars
    for bar, rate in zip(bars1, convergence_rates):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 2: Cost distribution
    if param_costs:
        cost_data = [param_costs[p] for p in params if p in param_costs and param_costs[p]]
        cost_labels = [p for p in params if p in param_costs and param_costs[p]]

        if cost_data:
            ax2.boxplot(cost_data, labels=cost_labels)
            ax2.set_title("Computational Cost Distribution")
            ax2.set_ylabel("Total Cost")
            ax2.tick_params(axis="x", rotation=45)

    # Plot 3: Most frequent optimal parameters (for first two parameters)
    subplot_idx = 2
    for param in list(optimal_param_counts.keys())[:2]:
        if optimal_param_counts[param]:
            # Get top 10 most frequent values
            most_common = optimal_param_counts[param].most_common(10)
            values, counts = zip(*most_common)

            ax = ax3 if subplot_idx == 2 else ax4
            bars = ax.bar(range(len(values)), counts, color="lightsteelblue")
            ax.set_title(f"Most Frequent Optimal {param.upper()} Values")
            ax.set_xlabel(f"{param.upper()} Value")
            ax.set_ylabel("Frequency")

            # Set x-axis labels
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([str(v) for v in values], rotation=45, ha="right")

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, str(count), ha="center", va="bottom")

            subplot_idx += 1

    # If we have fewer than 2 parameters with optimal counts, fill remaining subplots
    if len(optimal_param_counts) < 2:
        if not optimal_param_counts:
            ax3.text(
                0.5,
                0.5,
                "No converged\noptimal parameters",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
            )
            ax3.set_title("Optimal Parameter Frequencies")

        ax4.text(
            0.5,
            0.5,
            "Additional statistics\nnot available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=12,
        )
        ax4.set_title("Additional Statistics")

    plt.tight_layout()
    plt.savefig("burgers_1d_dummy_statistics.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    total_tasks = sum(result["total_tasks"] for result in all_results)
    total_converged = sum(
        sum(1 for task in result["results"] if task.get("converged", False)) for result in all_results
    )

    print(f"Total tasks executed: {total_tasks}")
    print(f"Overall convergence rate: {total_converged/max(total_tasks, 1):.1%} ({total_converged}/{total_tasks})")

    for param in param_convergence_rates:
        stats = param_convergence_rates[param]
        rate = stats["converged"] / max(stats["total"], 1)
        print(f"{param.upper()} parameter: {rate:.1%} ({stats['converged']}/{stats['total']})")

    if param_costs:
        print(f"\nAverage computational costs:")
        for param, costs in param_costs.items():
            if costs:
                avg_cost = sum(costs) / len(costs)
                print(f"  {param.upper()}: {avg_cost:.1f}")

    print("\nMost frequent optimal parameters:")
    for param, counter in optimal_param_counts.items():
        if counter:
            most_common_val, most_common_count = counter.most_common(1)[0]
            total_optimal = sum(counter.values())
            frequency = most_common_count / total_optimal
            print(f"  {param.upper()}: {most_common_val} (appears {frequency:.1%} of the time)")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Burgers 1D Dummy Solution Generation")
    print("=" * 60)

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "burgers_1d_config.yaml")
    config = load_config(config_path)

    # Build target configurations
    target_configs = build_target_configs(config)

    # Print overview
    total_tasks = sum(len(profiles) * len(precision_levels) for profiles, precision_levels in target_configs.values())
    print(f"Total tasks to execute: {total_tasks}")
    print(f"Target parameters: {list(target_configs.keys())}")
    print(f"Profiles: {config['profiles']['active_profiles']}")
    print(f"Precision levels: {list(config['precision_levels'].keys())}")

    # Execute all tasks
    all_results = []

    for target_param, (profiles, precision_levels) in target_configs.items():
        param_config = config["target_parameters"][target_param]

        task_results = []
        for profile in profiles:
            for precision_level in precision_levels:
                precision_config = config["precision_levels"][precision_level]

                result = run_single_task(target_param, param_config, profile, precision_level, precision_config)
                task_results.extend(result)

        all_results.append({"target_param": target_param, "results": task_results, "total_tasks": len(task_results)})

    # Generate statistics and plots
    plot_statistics(all_results)

    print(f"\n{'='*60}")
    print("Dummy solution generation completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
