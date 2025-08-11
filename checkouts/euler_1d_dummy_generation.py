import itertools
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dummy_sols.euler_1d import find_convergent_cfl, find_convergent_n_space, find_optimal_beta, find_optimal_k
from checkouts.config_utils import load_config, build_target_configs


def plot_statistics(statistics, output_dir):
    """Plot convergence and optimal parameter statistics"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Convergence rate by precision level and target parameter
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Euler 1D Dummy Search Statistics", fontsize=16)

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

    # Plot 3: Optimal parameter frequency (for 0-shot tasks)
    ax = axes[1, 0]
    if statistics["optimal_k_values"]:
        k_values, k_counts = np.unique(list(statistics["optimal_k_values"]), return_counts=True)
        ax.bar([str(k) for k in k_values], k_counts, alpha=0.7, label="k parameter")

    if statistics["optimal_beta_values"]:
        beta_values, beta_counts = np.unique(list(statistics["optimal_beta_values"]), return_counts=True)
        ax.bar([str(b) for b in beta_values], beta_counts, alpha=0.7, label="beta parameter")

    ax.set_ylabel("Frequency")
    ax.set_title("Optimal Parameter Values (0-shot tasks)")
    ax.legend()

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
    plt.savefig(os.path.join(output_dir, "euler_1d_statistics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Create detailed statistics file
    stats_file = os.path.join(output_dir, "euler_1d_statistics_summary.txt")
    with open(stats_file, "w") as f:
        f.write("=== Euler 1D Dummy Search Statistics Summary ===\n\n")

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

        f.write("4. Optimal Parameter Frequencies (0-shot tasks):\n")
        if statistics["optimal_k_values"]:
            k_values, k_counts = np.unique(list(statistics["optimal_k_values"]), return_counts=True)
            f.write("   k parameter:\n")
            for k, count in zip(k_values, k_counts):
                f.write(f"     k={k}: {count} times\n")

        if statistics["optimal_beta_values"]:
            beta_values, beta_counts = np.unique(list(statistics["optimal_beta_values"]), return_counts=True)
            f.write("   beta parameter:\n")
            for b, count in zip(beta_values, beta_counts):
                f.write(f"     beta={b}: {count} times\n")


def main():
    print("=== Euler 1D Dummy Solution Generation ===")
    print("Loading configuration from euler_1d_config.yaml...")

    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "euler_1d_config.yaml")
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
        "optimal_k_values": [],
        "optimal_beta_values": [],
    }

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
                    if target_param == "cfl":
                        is_converged, best_param, cost_history, param_history = find_convergent_cfl(
                            profile=profile,
                            cfl=target_config["initial_value"],
                            beta=task_params["beta"],
                            k=task_params["k"],
                            n_space=task_params["n_space"],
                            tolerance_rmse=precision_vals["tolerance_rmse"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )

                    elif target_param == "n_space":
                        is_converged, best_param, cost_history, param_history = find_convergent_n_space(
                            profile=profile,
                            cfl=task_params["cfl"],
                            n_space=target_config["initial_value"],
                            beta=task_params["beta"],
                            k=task_params["k"],
                            tolerance_rmse=precision_vals["tolerance_rmse"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )

                    elif target_param == "beta":
                        is_converged, optimal_param, cost_history, param_history = find_optimal_beta(
                            profile=profile,
                            cfl=task_params["cfl"],
                            k=task_params["k"],
                            n_space=task_params["n_space"],
                            tolerance_rmse=precision_vals["tolerance_rmse"],
                            search_range_min=target_config["search_range_min"],
                            search_range_max=target_config["search_range_max"],
                            search_range_slice_num=target_config["search_range_slice_num"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        best_param = optimal_param[0] if optimal_param[0] is not None else None
                        if best_param is not None:
                            statistics["optimal_beta_values"].append(best_param)

                    elif target_param == "k":
                        is_converged, optimal_param, cost_history, param_history = find_optimal_k(
                            profile=profile,
                            cfl=task_params["cfl"],
                            beta=task_params["beta"],
                            n_space=task_params["n_space"],
                            tolerance_rmse=precision_vals["tolerance_rmse"],
                            search_range_min=target_config["search_range_min"],
                            search_range_max=target_config["search_range_max"],
                            search_range_slice_num=target_config["search_range_slice_num"],
                            multiplication_factor=target_config["multiplication_factor"],
                            max_iteration_num=target_config["max_iteration_num"],
                        )
                        best_param = optimal_param[0] if optimal_param[0] is not None else None
                        if best_param is not None:
                            statistics["optimal_k_values"].append(best_param)

                    # Update statistics
                    total_cost = sum(cost_history) if cost_history else 0
                    statistics["total_tasks"] += 1
                    statistics["convergence_by_precision"][precision_name]["total"] += 1
                    statistics["convergence_by_target"][target_param]["total"] += 1
                    statistics["convergence_by_target"][target_param]["costs"].append(total_cost)

                    if is_converged:
                        statistics["total_converged"] += 1
                        statistics["convergence_by_precision"][precision_name]["converged"] += 1
                        statistics["convergence_by_target"][target_param]["converged"] += 1
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
