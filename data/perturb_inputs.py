import numpy as np
import argparse

Z_EFF_VALUE = 2

KSUFFIX_REPLACEMENT_DICT = {
    '_4': '_3'
}

KV_REPLACEMENT_DICT = {
    'N_SPECIES': 3,
    'COLLISION_MODEL': 1,
    'Z_EFF_METHOD': 1,
    'Z_EFF': Z_EFF_VALUE
}

DELETE_KSUFFIX = [
    '_3'
]

PERTURBATION_KEYS = [
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
    'TEMP_1',
    'DLNNDR_1',
    'DLNTDR_1',
    'TEMP_2',
    'DLNNDR_2',
    'DLNTDR_2',
    'TEMP_3',
    'DLNNDR_3', 
    'DLNTDR_3',
]

# Remove 3rd ion (change NS=3)
# N4 becomes N3 (for all fields)
# Remove all *_3, replace all *_4 with *_3
def replace_ion3(input_dict):
    new_dict = input_dict.copy()

    # Delete keys by suffix
    for ksuffix in DELETE_KSUFFIX:
        for key, _ in input_dict.items():
            if ksuffix in key:
                del new_dict[key]
    
    # Replace keys
    for key, new_val in KV_REPLACEMENT_DICT.items():
        new_dict[key] = new_val

    # Replace keys by suffix
    for ksuffix, new_ksuffix in KSUFFIX_REPLACEMENT_DICT.items():
        for key, value in input_dict.items():
            if ksuffix in key:
                new_key = key.split('_')[0] + new_ksuffix
                new_dict[new_key] = value
                del new_dict[key]

    # print(new_dict)
    return new_dict

# Replaces input_file in-place with perturbed inputs, saves old input file to temp_file
# Computes the error between perturbed inputs and stores as npy array
def perturb_inputs(input_file, temp_file, error_file, original_input_file):
    input_dict = get_input_dict(original_input_file)
    input_dict_no_ion3 = replace_ion3(input_dict)
    input_dict_qn = enforce_quasineutrality(input_dict_no_ion3)

    # Rewrite input file after replacing ion3 and enforcing QN
    update_input_file(input_file, temp_file, input_dict_qn)

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

def enforce_quasineutrality(input_dict):
    dens_3 = input_dict['DENS_3']
    dens_1 = dens_3 * (6 - Z_EFF_VALUE) / 5
    dens_2 = dens_3 * (Z_EFF_VALUE - 1) / 30

    input_dict['DENS_1'] = dens_1
    input_dict['DENS_2'] = dens_2
    return input_dict

def compute_perturbations(perturbation_keys, input_dict, error_file, mean=0, std=0.04, clamp=0.10):
    perturbation_dict = {}
    errors = []
    print(f'Relative errors after perturbations from Normal with mean={mean}, std={std}: ' + ("=" * 30))
    for key in perturbation_keys:
        old_value = input_dict[key]
        perturbation = np.random.normal(loc=mean, scale=std)
        # Clamp perturbation
        perturbation = min(perturbation, clamp)

        new_value = old_value + (old_value * perturbation)

        epsilon = 1e-12
        error = np.abs(np.abs(old_value - new_value) / (old_value + epsilon))
        errors.append(error)
        print(f'{key}: {error * 100}%')

        perturbation_dict[key] = new_value
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

def update_input_file(input_file, temp_file, dict):
    try:
        with open(input_file, 'r') as file:
            lines = []
            updated_lines = []

            for line in file:
                lines.append(line)
            
            for key, value in dict.items():
                updated_lines.append(f'{key}={value}\n') 
                    
        with open(input_file, 'w') as f:
            f.writelines(updated_lines)

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
