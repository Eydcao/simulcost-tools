import os
import yaml


def normalize_numeric_values(obj):
    """
    Recursively convert string representations of numbers (especially scientific notation)
    to appropriate numeric types (int, float).

    Args:
        obj: Any object that may contain string numeric values

    Returns:
        The same object with string numbers converted to numeric types
    """
    if isinstance(obj, dict):
        return {k: normalize_numeric_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_numeric_values(item) for item in obj]
    elif isinstance(obj, str):
        # Try to convert string to numeric value
        obj_lower = obj.lower()
        try:
            # Handle scientific notation (e.g., "1e-8", "2.5e+3")
            if "e" in obj_lower and any(c.isdigit() for c in obj):
                return float(obj)
            # Handle decimal numbers
            elif "." in obj and obj.replace(".", "").replace("-", "").isdigit():
                return float(obj)
            # Handle integers
            elif obj.replace("-", "").isdigit():
                return int(obj)
            else:
                return obj  # Keep as string if not numeric
        except ValueError:
            return obj  # Keep as string if conversion fails
    else:
        return obj  # Return unchanged for other types


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Normalize any string numeric values to proper numeric types
    config = normalize_numeric_values(config)

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
            "search_range_min": param_info.get("search_range_min"),
            "search_range_max": param_info.get("search_range_max"),
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
