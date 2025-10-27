import numpy as np
import re
from pathlib import Path

# Fix np random seed for reproducibility
np.random.seed(42)


def create_hasegawa_profiles(case_configs, base_profile_path):
    """
    Create multiple Hasegawa-Mima profiles based on the p1 base profile.
    Preserves comments and formatting from the base profile.

    Args:
        case_configs: List of dictionaries with case configurations (case)
        base_profile_path: Path to the base p1.yaml file

    Returns:
        List of paths to the generated config files
    """
    base_path = Path(base_profile_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base profile not found at {base_path}")

    with open(base_path, "r") as f:
        lines = f.readlines()

    generated_paths = []
    num_profiles = len(case_configs)

    for i in range(num_profiles):
        profile_name = f"p{i+2}"
        config = case_configs[i]

        update_params = {
            "case": config["case"],
            "dump_dir": f"sim_res/hasegawa_mima_linear/{profile_name}",
        }

        new_lines = []
        for line in lines:
            if not line.strip() or (line.strip().startswith("#") and not re.match(r"^(\s*)([a-zA-Z_]+):", line)):
                new_lines.append(line)
                continue

            param_match = re.match(r"^(\s*)([a-zA-Z_]+):\s*(.*?)(\s*#.*)?$", line)
            if param_match:
                spaces, param_name, current_value, comment = param_match.groups()
                if comment is None:
                    comment = ""

                if param_name in update_params:
                    if isinstance(update_params[param_name], str):
                        new_value = f'"{update_params[param_name]}"'
                    else:
                        new_value = str(update_params[param_name])

                    new_line = f"{spaces}{param_name}: {new_value}{comment}\n"
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        output_dir = base_path.parent
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{profile_name}.yaml"

        with open(output_path, "w") as f:
            f.writelines(new_lines)

        generated_paths.append(output_path)
        print(f"Created profile {profile_name} at {output_path} with case '{config['case']}'")

    return generated_paths


if __name__ == "__main__":
    # Create Hasegawa-Mima profiles for several initial condition cases
    # Note: p1 = monopole, p3 = sinusoidal (separate, not generated here)
    case_configs = [
        # {"case": "monopole"}, # p1 is this
        {"case": "dipole"},  # p2
        # {"case": "sinusoidal"},  # p3 (moved out, separate config)
        {"case": "sin_x_gauss_y"},  # p4 (was p4, now p3 in generated files)
        {"case": "gauss_x_sin_y"},  # p5 (was p5, now p4 in generated files)
    ]

    profiles = create_hasegawa_profiles(
        case_configs=case_configs, base_profile_path="./run_configs/hasegawa_mima_linear/p1.yaml"
    )
