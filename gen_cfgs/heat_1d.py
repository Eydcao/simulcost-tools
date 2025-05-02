import numpy as np
import re
from pathlib import Path

# fix np random seed
np.random.seed(42)

def create_heat1d_profiles(num_profiles, base_profile_path):
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
        # Heat transfer coefficient (log-uniform between 0.1 and 100)
        log_h_min = np.log10(0.1)
        log_h_max = np.log10(100)
        log_h = np.random.uniform(log_h_min, log_h_max)

        # Create parameter dictionary with random values
        random_params = {
            "h": round(10**log_h, 2),  # Heat transfer coefficient [W/m²-K]
            "L": round(np.random.uniform(0.1, 0.2), 3),  # Rod length [m]
            "k": round(np.random.uniform(0.5, 1), 2),  # Thermal conductivity [W/m-K]
            "rho": round(np.random.uniform(1000, 2000)),  # Density [kg/m³]
            "cp": round(np.random.uniform(800, 1000)),  # Specific heat [J/kg-K]
            "T_inf": round(np.random.uniform(4, 20)),  # Ambient temp [°C]
            "T_init": round(np.random.uniform(21, 30)),  # Initial temp [°C]
            "record_dt": round(np.random.uniform(1, 8)) * 10,  # Recording interval [s]
        }

        # Add dump_dir with new profile name
        random_params["dump_dir"] = f"sim_res/heat_1d/{profile_name}"

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
    random_profiles = create_heat1d_profiles(num_profiles=9, base_profile_path="./run_configs/heat_1d/p1.yaml")
