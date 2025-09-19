import numpy as np
import re
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def create_ns_channel_profiles(num_profiles_per_bc, base_profile_path, solver_name):
    base_path = Path(base_profile_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base profile not found at {base_path}")

    # Read original config lines
    with open(base_path, "r") as f:
        lines = f.readlines()

    generated_paths = []
    
    # Define boundary conditions - generate num_profiles_per_bc for each
    boundary_conditions = ['channel_flow', 'back_stair_flow', 'expansion_channel', 'cube_driven_flow']
    
    profile_counter = 2  # Start from p2 (p1 is the base profile)

    for boundary_condition in boundary_conditions:
        for j in range(num_profiles_per_bc):
            profile_name = f"p{profile_counter}"
            profile_counter += 1
            
            # Randomized environment-dependent parameters
            random_params = {
                "length": round(np.random.uniform(10.0, 30.0), 2),
                "breadth": round(np.random.uniform(0.5, 2.0), 2),
                "mu": round(np.random.uniform(0.005, 0.05), 5),
                "rho": round(np.random.uniform(0.8, 5.0), 2),
                "boundary_condition": f"\"{boundary_condition}\"",
                "dump_dir": f"\"sim_res/{solver_name}/{profile_name}\""
            }
        
            if random_params["boundary_condition"] != "\"channel_flow\"":
                # Add cfg.other_params for non-channel flow conditions
                # First append the other_params line
                random_params["other_params"] = "\n"
                if random_params["boundary_condition"] == "\"expansion_channel\"":
                    random_params["other_params"] += f"  wall_height: {int(round(np.random.uniform(10, 20)))}\n"
                    random_params["other_params"] += f"  wall_width: {int(round(np.random.uniform(30, 100)))}\n"
                elif random_params["boundary_condition"] == "\"cube_driven_flow\"":
                    random_params["other_params"] += f"  wall_height: {int(round(np.random.uniform(5, 15)))}\n"
                    random_params["other_params"] += f"  wall_width: {int(round(np.random.uniform(5, 15)))}\n"
                    random_params["other_params"] += f"  wall_start_height: {int(round(np.random.uniform(10, 30)))}\n"
                    random_params["other_params"] += f"  wall_start_width: {int(round(np.random.uniform(70, 90)))}\n"
                elif random_params["boundary_condition"] == "\"back_stair_flow\"":
                    random_params["other_params"] += f"  wall_height: {int(round(np.random.uniform(5, 15)))}\n"
                    random_params["other_params"] += f"  wall_width: {int(round(np.random.uniform(5, 15)))}\n"

            new_lines = []
            for line in lines:
                if not line.strip() or line.strip().startswith("#"):
                    new_lines.append(line)
                    continue

                match = re.match(r"^(\s*)([a-zA-Z_]+):\s*(.*?)(\s*#.*)?$", line)
                if match:
                    spaces, key, _, comment = match.groups()
                    if comment is None:
                        comment = ""
                    if key in random_params:
                        new_value = random_params[key]
                        new_line = f"{spaces}{key}: {new_value}{comment}\n"
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            # Append other_params if boundary_condition is not "channel_flow"
            if random_params["boundary_condition"] != "\"channel_flow\"" and "other_params" in random_params:
                new_lines.append(f"\nother_params: {random_params['other_params']}\n")

            output_path = base_path.parent / f"{profile_name}.yaml"
            with open(output_path, "w") as f:
                f.writelines(new_lines)

            generated_paths.append(str(output_path))
            print(f"Generated {profile_name} ({boundary_condition}) at {output_path}")

    return generated_paths

# Example usage:
if __name__ == "__main__":
    create_ns_channel_profiles(num_profiles_per_bc=5, base_profile_path="../run_configs/ns_channel_2d/p1.yaml", solver_name="ns_channel_2d")