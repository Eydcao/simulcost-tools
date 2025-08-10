#!/usr/bin/env python3
"""
Euler 1D Dummy Solution Generation Script
Based on the configuration in euler_1d_config.yaml

This script reads the centralized YAML configuration and generates all dummy solution 
tasks for the Euler 1D solver following the specified strategy and parameters.
"""

import subprocess
import itertools
import os
import yaml


def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), "euler_1d_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def build_target_configs(config):
    """Build target configurations from YAML config - normalize single values to lists"""
    target_configs = {}

    for param_name, param_info in config["target_parameters"].items():
        target_config = {
            "search_type": param_info["search_type"],
            "initial_value": param_info.get("initial_value"),
            "non_target_parameters": {},
            # Search parameters
            "multiplication_factor": param_info.get("multiplication_factor"),
            "max_iteration_num": param_info.get("max_iteration_num"),
            "search_range_min": (
                param_info.get("search_range", [None, None])[0]
                if isinstance(param_info.get("search_range"), list)
                else None
            ),
            "search_range_max": (
                param_info.get("search_range", [None, None])[1]
                if isinstance(param_info.get("search_range"), list)
                else None
            ),
            "search_range_slice_num": param_info.get("search_range_slice_num"),
        }

        # Normalize all non-target parameters to lists for consistent iteration
        for key, value in param_info["non_target_parameters"].items():
            if isinstance(value, list):
                target_config["non_target_parameters"][key] = value
            else:
                target_config["non_target_parameters"][key] = [value]  # Convert single value to list

        target_configs[param_name] = target_config

    return target_configs


def main():
    print("=== Euler 1D Dummy Solution Generation ===")
    print("Loading configuration from euler_1d_config.yaml...")

    # Load configuration from YAML
    try:
        config = load_config()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return

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

    # Total task counter
    total_tasks = 0

    # Change to repository root directory
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)
    print(f"Working directory: {os.getcwd()}")

    # Generate all task combinations
    for precision_name, precision_vals in precision_configs.items():
        print(f"\n--- Processing {precision_name.upper()} precision ---")
        precision_tasks = 0

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
                    # Build command
                    cmd_parts = [
                        "python",
                        "dummy_sols/euler_1d.py",
                        f"--profile",
                        profile,
                        f"--task",
                        target_param,
                        f"--tolerance_rmse",
                        str(precision_vals["tolerance_rmse"]),
                    ]

                    # Add target parameter initial value if it's an iterative parameter
                    if target_config["initial_value"] is not None:
                        cmd_parts.extend([f"--{target_param}", str(target_config["initial_value"])])

                    # Add all non-target parameters
                    for param_name, param_value in zip(param_names, combination):
                        cmd_parts.extend([f"--{param_name}", str(param_value)])

                    # Add search parameters for the target parameter
                    if target_config["multiplication_factor"] is not None:
                        cmd_parts.extend(["--multiplication_factor", str(target_config["multiplication_factor"])])
                    if target_config["max_iteration_num"] is not None:
                        cmd_parts.extend(["--max_iteration_num", str(target_config["max_iteration_num"])])
                    if target_config["search_range_min"] is not None:
                        cmd_parts.extend(["--search_range_min", str(target_config["search_range_min"])])
                    if target_config["search_range_max"] is not None:
                        cmd_parts.extend(["--search_range_max", str(target_config["search_range_max"])])
                    if target_config["search_range_slice_num"] is not None:
                        cmd_parts.extend(["--search_range_slice_num", str(target_config["search_range_slice_num"])])

                    # Execute command
                    print(f"      Running: {' '.join(cmd_parts)}")
                    try:
                        result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=1800)
                        if result.returncode != 0:
                            print(f"      ERROR: {result.stderr}")
                        else:
                            print(f"      SUCCESS")
                        total_tasks += 1
                        precision_tasks += 1
                    except subprocess.TimeoutExpired:
                        print(f"      TIMEOUT: Task took longer than 30 minutes")
                    except Exception as e:
                        print(f"      EXCEPTION: {e}")

        print(f"  {precision_name.capitalize()} precision tasks: {precision_tasks}")

    print(f"\n=== Generation Complete ===")
    print(f"Total tasks generated: {total_tasks}")
    print("All Euler 1D dummy solutions cached and ready for LLM automation!")

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
    print(f"  Expected total: {expected_total}")
    print(f"  Actual total: {total_tasks}")


if __name__ == "__main__":
    main()
