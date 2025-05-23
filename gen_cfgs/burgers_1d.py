import numpy as np
import re
from pathlib import Path

# Fix np random seed
np.random.seed(42)


def create_burgers1d_profiles(initial_conditions, base_profile_path):
    """
    Create multiple random Burgers1D profiles based on the p1 base profile.
    Preserves comments and formatting from the base profile.

    Args:
        initial_conditions: initial conditions to use for the profiles
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

    num_profiles = len(initial_conditions)  # p1 + random profiles

    # Generate random profiles
    for i in range(num_profiles):
        profile_name = f"p{i+2}"  # p2, p3, etc.

        # Generate random parameters
        random_params = {"case": initial_conditions[i]}

        # Add dump_dir with new profile name
        random_params["dump_dir"] = f"sim_res/burgers_1d/{profile_name}"

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

                # Update parameter if it's in our random_params dictionary
                if param_name in random_params:
                    # Format the new value similar to the original (preserve quotes if needed)
                    if isinstance(random_params[param_name], str):
                        new_value = f'"{random_params[param_name]}"'
                    else:
                        new_value = str(random_params[param_name])

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
        print(f"Created profile {profile_name} at {output_path} with preserved formatting")

    return generated_paths


if __name__ == "__main__":
    # Create multiple random profiles (BUT FOR SIN)
    ic = ["rarefaction", "sod", "double_shock", "blast"]
    random_profiles = create_burgers1d_profiles(
        initial_conditions=ic, base_profile_path="./run_configs/burgers_1d/p1.yaml"
    )
