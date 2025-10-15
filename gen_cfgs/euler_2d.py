import numpy as np
import re
from pathlib import Path

# Fix np random seed
np.random.seed(42)


def create_euler2d_profiles(case_configs, base_profile_path):
    """
    Create multiple Euler2D profiles based on the p1 base profile.
    Preserves comments and formatting from the base profile.

    Args:
        case_configs: List of dictionaries with case configurations
                     (testcase, end_frame, record_dt, n_grid_x)
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

    num_profiles = len(case_configs)  # Additional profiles beyond p1

    # Generate profiles
    for i in range(num_profiles):
        profile_name = f"p{i+2}"  # p2, p3, etc.

        # Get configuration for this profile
        config = case_configs[i]

        # Set parameters to update
        update_params = {
            "testcase": config["testcase"],
            "n_grid_x": config["n_grid_x"],
            "cfl": config.get("cfl", 0.5),
            "cg_tolerance": config.get("cg_tolerance", 1.0e-7),
            "end_frame": config["end_frame"],
            "record_dt": config["record_dt"],
            "dump_dir": f"sim_res/euler_2d/{profile_name}",
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
        print(
            f"Created profile {profile_name} at {output_path} with testcase={config['testcase']}, "
            f"end_frame={config['end_frame']}, record_dt={config.get('record_dt', 'default')}, n_grid_x={config['n_grid_x']}"
        )

    return generated_paths


if __name__ == "__main__":
    # Create Euler 2D profiles for different test cases
    # Each profile represents a unique physical scenario
    # Resolution (n_grid_x) is a tunable parameter that should be optimized
    #
    # Testcases:
    # 0 = central_explosion (aspect_ratio=1.0)
    # 1 = stair_flow (aspect_ratio=1/3)
    # 2 = cylinder_gravity (aspect_ratio=1/3)
    # 3 = mach_diamond (aspect_ratio=1/2)

    case_configs = [
        # p2: stair_flow test case
        # Original: 50 frames * (1/360 s/frame) = 0.139s ≈ 0.14s
        # New: 20 frames * 0.007 s/frame = 0.14s
        {"testcase": 1, "end_frame": 20, "record_dt": 0.007, "n_grid_x": 64, "cfl": 0.5, "cg_tolerance": 1.0e-7},

        # p3: cylinder_gravity test case
        # Original: 100 frames * (1/360 s/frame) = 0.278s ≈ 0.3s
        # New: 20 frames * 0.015 s/frame = 0.3s
        {"testcase": 2, "end_frame": 20, "record_dt": 0.015, "n_grid_x": 64, "cfl": 0.5, "cg_tolerance": 1.0e-7},

        # p4: mach_diamond test case
        # Original: 120 frames * (1/180 s/frame) = 0.667s ≈ 0.7s
        # New: 20 frames * 0.035 s/frame = 0.7s
        {"testcase": 3, "end_frame": 20, "record_dt": 0.035, "n_grid_x": 64, "cfl": 0.5, "cg_tolerance": 1.0e-7},
    ]

    # Generate the configs
    base_profile = Path(__file__).parent.parent / "run_configs" / "euler_2d" / "p1.yaml"
    generated_paths = create_euler2d_profiles(case_configs, base_profile)

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(generated_paths)} profile configurations:")
    for path in generated_paths:
        print(f"  - {path}")
    print(f"{'='*60}")
    print(f"\nNote: n_grid_x values in configs are defaults.")
    print(f"Resolution is a tunable parameter that should be optimized by the dummy solution.")
