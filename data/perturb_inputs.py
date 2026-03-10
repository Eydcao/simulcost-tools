import numpy as np
import argparse

Z_EFF_VALUE = 1.8 # Updated 

KV_ADD_DICT = {
    'Z_EFF_METHOD': 2,      # Updated
    # 'Z_EFF': Z_EFF_VALUE  # Updated, ZEFF will automatically be given based on density values
}
KV_REPLACEMENT_DICT = {
    'N_SPECIES': 3,
    'COLLISION_MODEL': 4,
}

PERTURBATION_KEYS = [
    'BETAE_UNIT',
    'SHIFT',
    'Q',
    'S',
    'KAPPA',
    'S_KAPPA',
    'DELTA',
    'S_DELTA',
    'ZETA',
    'S_ZETA',
    'GAMMA_P',
    'MACH',
    # 'TEMP_1',
    'DLNNDR_1',
    'DLNTDR_1',
    # 'TEMP_2', # If perturb, match to TEMP_1
    'DLNNDR_2',
    'DLNTDR_2',
    # 'TEMP_3', # Do not perturb, needs to remain 1 for normalization scaling
    'DLNNDR_3', 
    'DLNTDR_3',
]

# Remove 3rd ion (change NS=3)
# N4 becomes N3 (for all fields)
# Remove all *_3, replace all *_4 with *_3
def replace_ion3(input_file, output_file):
    try:
        with open(input_file, 'r') as file:
            lines = []
            updated_lines = []

            for line in file:
                lines.append(line)
            for line in lines:
                foundline = False
                for key, value in KV_REPLACEMENT_DICT.items():
                    if key in line:
                        new_line = f'{key}={value}\n'
                        updated_lines.append(new_line)
                        foundline = True
                        break
                if ('_3' not in line) and ('_4' not in line) and (not foundline):
                    updated_lines.append(line)
                if '_4' in line:
                    new_line = line.split('_4')[0] + '_3' + line.split('_4')[1]
                    updated_lines.append(new_line)
            
            for key, value in KV_ADD_DICT.items():
                new_line = f'{key}={value}\n'
                updated_lines.append(new_line)
                    
        with open(output_file, 'w') as f:
            f.writelines(updated_lines)
            
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 'NU_EE' : Collisionality (b/w 0.05 - 0.8) can sample uniformly
def sample_collisionality(input_file, low=0.05, high=0.8):
    nu_ee = np.random.uniform(low, high)
    try:
        with open(input_file, 'r') as file:
            lines = []
            updated_lines = []

            for line in file:
                lines.append(line)

            for line in lines:
                if 'NU_EE' in line:
                    new_line = f'NU_EE={nu_ee}\n'
                    updated_lines.append(new_line)
                else:
                    updated_lines.append(line)
                    
        with open(input_file, 'w') as f:
            f.writelines(updated_lines)
            
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def enforce_quasineutrality(input_file, input_dict):
    dens_3 = input_dict['DENS_3']
    dens_1 = dens_3 * (6 - Z_EFF_VALUE) / 5
    dens_2 = dens_3 * (Z_EFF_VALUE - 1) / 30
    try:
        with open(input_file, 'r') as file:
            lines = []
            updated_lines = []

            for line in file:
                lines.append(line)

            for line in lines:
                if 'DENS_1' in line:
                    new_line = f'DENS_1={dens_1}\n'
                    updated_lines.append(new_line)
                elif 'DENS_2' in line:
                    new_line = f'DENS_2={dens_2}\n'
                    updated_lines.append(new_line)
                else:
                    updated_lines.append(line)
                    
        with open(input_file, 'w') as f:
            f.writelines(updated_lines)
            
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replaces input_file in-place with perturbed inputs, saves old input file to temp_file
# Computes the error between perturbed inputs and stores as npy array
def perturb_inputs(input_file, temp_file, error_file, original_input_file):
    replace_ion3(original_input_file, input_file)
    input_dict_no_ion3 = get_input_dict(input_file)

    enforce_quasineutrality(input_file, input_dict_no_ion3)
    sample_collisionality(input_file)
    input_dict_qn = get_input_dict(input_file)

    # Apply perturbations to updated input file
    perturbation_dict = compute_perturbations(PERTURBATION_KEYS, input_dict_qn, error_file)
    apply_perturbations(input_file, temp_file, perturbation_dict)

def get_input_dict(input_file):
    try:
        with open(input_file, 'r') as file:
            input_dict = {}
            for line in file:
                if '=' in line:
                    key = line.split('=')[0].strip()
                    value = float(line.split('=')[1].strip())
                    input_dict[key] = value
            return input_dict
            
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

def compute_perturbations(perturbation_keys, input_dict, error_file, mean=0, std=0.08, clamp=0.20):
    perturbation_dict = {}
    errors = []
    print(f'Relative errors after perturbations from Normal with mean={mean}, std={std}: ' + ("=" * 30))
    for key in perturbation_keys:
        old_value = input_dict[key]
        perturbation_factor = np.random.normal(loc=mean, scale=std)
        # Clamp perturbation
        if np.abs(perturbation_factor) > clamp:
            perturbation_factor = (perturbation_factor * clamp) / np.abs(perturbation_factor)

        perturbation = perturbation_factor * old_value

        epsilon = 1e-12
        error = np.abs(np.abs(old_value - (old_value + perturbation)) / (old_value + epsilon))
        errors.append(error)
        print(f'{key}: {error * 100}%')

        perturbation_dict[key] = perturbation
    errors = np.stack(errors, axis=0)
    np.save(error_file, errors)
    return perturbation_dict

def apply_perturbations(input_file, temp_file, perturbation_dict):
    try:
        with open(input_file, 'r') as file:
            lines = []
            perturbed_lines = []
            for line in file:
                lines.append(line)
            
            for line in lines:
                key_found = False
                for key, perturbation in perturbation_dict.items():
                    if line.split('=')[0].strip() == key:
                        old_value = float(line.split('=')[1].strip())
                        new_value = old_value + perturbation
                        pline = f'{key}={new_value}\n'
                        perturbed_lines.append(pline)
                        key_found = True
                        break
                if not key_found:
                    perturbed_lines.append(line)
                    
        with open(input_file, 'w') as f:
            f.writelines(perturbed_lines)

        with open(temp_file, 'w') as f:
            f.writelines(lines)
            
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file")
    parser.add_argument("-t", "--temp_file")
    parser.add_argument("-e", "--error_file")
    parser.add_argument("-o", "--original_file")
    args = parser.parse_args()

    perturb_inputs(args.input_file, args.temp_file, args.error_file, args.original_file)
