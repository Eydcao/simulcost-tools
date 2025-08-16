import numpy as np
import re
from pathlib import Path

# Fix np random seed
np.random.seed(42)


def create_heat_steady_2d_profiles(boundary_configs, base_profile_path):
    """
    Create multiple Heat Steady 2D profiles based on the p1 base profile.
    Preserves comments and formatting from the base profile.

    Args:
        boundary_configs: List of dictionaries with boundary condition configurations
        base_profile_path: Path to the base p1.yaml file

    Returns:
        List of paths to the generated config files
    """
    # Check if base profile exists
    base_path = Path(base_profile_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base profile not found at {base_path}")

    # Read the base profile as text
    with open(base_path, "r") as f:
        lines = f.readlines()

    generated_paths = []

    num_profiles = len(boundary_configs)

    # Generate profiles
    for i in range(num_profiles):
        profile_name = f"p{i+2}"  # p2, p3, etc.

        # Get configuration for this profile
        config = boundary_configs[i]

        # Set parameters to update
        update_params = {
            "T_top": config["T_top"],
            "T_bottom": config["T_bottom"],
            "T_left": config["T_left"],
            "T_right": config["T_right"],
            "T_init": config.get("T_init", 0.0),
            "dump_dir": f"sim_res/heat_steady_2d/{profile_name}",
        }

        # Modify the lines to update parameters while preserving comments and format
        new_lines = []
        for line in lines:
            # Skip empty lines and comments
            if not line.strip() or (line.strip().startswith("#") and not re.match(r"^(\s*)([a-zA-Z_]+):", line)):
                new_lines.append(line)
                continue

            # Check for parameters to update
            param_match = re.match(r"^(\s*)([a-zA-Z_]+):\s*(.*?)(\s*#.*)?$", line)
            if param_match:
                spaces, param_name, current_value, comment = param_match.groups()
                if comment is None:
                    comment = ""

                # Update parameter if it's in our update_params dictionary
                if param_name in update_params:
                    # Format the new value similar to the original (preserve quotes if needed)
                    if isinstance(update_params[param_name], str):
                        new_value = f'"{update_params[param_name]}"'
                    else:
                        new_value = str(update_params[param_name])

                    # Construct the new line with same spacing and comments
                    new_line = f"{spaces}{param_name}: {new_value}{comment}\n"
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Save to a new file
        output_dir = base_path.parent
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{profile_name}.yaml"

        with open(output_path, "w") as f:
            f.writelines(new_lines)

        generated_paths.append(output_path)
        print(f"Created profile {profile_name} at {output_path} with boundary conditions: {config}")

    return generated_paths


if __name__ == "__main__":
    # Create Heat Steady 2D profiles with diverse boundary condition patterns
    boundary_configs = [
        # p2: Hot sides opposite (top and bottom hot)
        {"T_top": 1.0, "T_bottom": 1.0, "T_left": 0.0, "T_right": 0.0, "T_init": 0.0},
        # p3: Corner hot spot (right side hot)
        {"T_top": 0.0, "T_bottom": 0.0, "T_left": 0.0, "T_right": 1.0, "T_init": 0.0},
        # p4: Alternating pattern (left hot, moderate top/bottom)
        {"T_top": 0.5, "T_bottom": 0.5, "T_left": 1.0, "T_right": 0.0, "T_init": 0.25},
        # p5: Uniform heating (all boundaries warm)
        {"T_top": 0.8, "T_bottom": 0.8, "T_left": 0.8, "T_right": 0.8, "T_init": 0.2},
    ]

    profiles = create_heat_steady_2d_profiles(
        boundary_configs=boundary_configs, base_profile_path="./run_configs/heat_steady_2d/p1.yaml"
    )
