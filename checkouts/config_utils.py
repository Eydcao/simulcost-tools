import os
import yaml


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def build_target_configs(config):
    target_configs = {}

    for param_name, param_info in config["target_parameters"].items():
        target_config = {
            "search_type": param_info["search_type"],
            "initial_value": param_info.get("initial_value"),
            "non_target_parameters": {},
            # Search parameters
            "multiplication_factor": param_info.get("multiplication_factor"),
            "max_iteration_num": param_info.get("max_iteration_num"),
            "search_range": param_info.get("search_range"),
            "search_range_slice_num": param_info.get("search_range_slice_num"),
            # Additional parameters for different search types
            "schedule_options": param_info.get("schedule_options"),
        }

        # Normalize all non-target parameters to lists for consistent iteration
        for key, value in param_info["non_target_parameters"].items():
            if isinstance(value, list):
                target_config["non_target_parameters"][key] = value
            else:
                target_config["non_target_parameters"][key] = [value]  # Convert single value to list

        target_configs[param_name] = target_config

    return target_configs
