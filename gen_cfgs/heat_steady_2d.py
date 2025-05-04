import numpy as np
import re
from pathlib import Path

# fix np random seed
np.random.seed(42)


def create_heat1d_profiles(num_profiles, base_profile_path, solver_name):
    """
    Create multiple random Heat1D profiles based on the p1 base profile.
    Preserves comments and formatting from the base profile.

    Args:
        num_profiles: Number of random profiles to generate
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

    # Generate random profiles
    for i in range(num_profiles):
        profile_name = f"p{i+2}"  # p2, p3, etc.

        # Generate random parameters

        # Create parameter dictionary with random values
        random_params = {
            "T_top": round(np.random.uniform(0.0, 1.0), 2),
            "T_bottom": round(np.random.uniform(0.0, 1.0), 2),
            "T_left": round(np.random.uniform(0.0, 1.0), 2),
            "T_right": round(np.random.uniform(0.0, 1.0), 2),
        }

        # Add dump_dir with new profile name
        random_params["dump_dir"] = f"sim_res/{solver_name}/{profile_name}"

        # Modify the lines to update parameters while preserving comments and format
        new_lines = []
        for line in lines:
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith("#"):
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
    # Example usage:

    # 1. Create multiple random profiles
    random_profiles = create_heat1d_profiles(
        num_profiles=9, base_profile_path="./run_configs/heat_steady_2d/p1.yaml", solver_name="heat_steady_2d"
    )
