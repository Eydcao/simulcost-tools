import numpy as np
import argparse

PERTURBATION_KEYS = [
    'SHIFT',
    'ZMAG',
    'DZMAG',
    'Q',
    'S',
    'KAPPA',
    'S_KAPPA',
    'DELTA',
    'S_DELTA',
    'ZETA',
    'S_ZETA',

    # Geometry (Advanced)
    'SHAPE_SIN3',
    'SHAPE_S_SIN3',
    'SHAPE_SIN4',
    'SHAPE_S_SIN4',
    'SHAPE_SIN5',
    'SHAPE_S_SIN5',
    'SHAPE_SIN6',
    'SHAPE_S_SIN6',
    'SHAPE_COS0',
    'SHAPE_S_COS0',
    'SHAPE_COS1',
    'SHAPE_S_COS1',
    'SHAPE_COS2',
    'SHAPE_S_COS2',
    'SHAPE_COS3',
    'SHAPE_S_COS3',
    'SHAPE_COS4',
    'SHAPE_S_COS4',
    'SHAPE_COS5',
    'SHAPE_S_COS5',
    'SHAPE_COS6',
    'SHAPE_S_COS6',

    # Rotation (Sonic)
    'GAMMA_E',
    'GAMMA_P',
    'MACH',

    # Species
    'MASS_1',
    'DENS_1',
    'TEMP_1',
    'DLNNDR_1',
    'DLNTDR_1',
    'SDLNNDR_1',
    'SDLNTDR_1',

    'MASS_2',
    'DENS_2',
    'TEMP_2',
    'DLNNDR_2',
    'DLNTDR_2',
    'SDLNNDR_2',
    'SDLNTDR_2',

    'MASS_3',
    'DENS_3',
    'TEMP_3',
    'DLNNDR_3',
    'DLNTDR_3',
    'SDLNNDR_3',
    'SDLNTDR_3',

    'MASS_4',
    'DENS_4',
    'TEMP_4',
    'DLNNDR_4', 
    'DLNTDR_4',
    'SDLNNDR_4',
    'SDLNTDR_4'
]
# Replaces input_file in-place with perturbed inputs, saves old input file to temp_file
# Computes the error between perturbed inputs and stores as npy array
def perturb_inputs(input_file, temp_file, error_file):
    input_dict = get_input_dict(input_file)
    perturbation_dict = compute_perturbations(PERTURBATION_KEYS, input_dict, error_file)
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

def compute_perturbations(perturbation_keys, input_dict, error_file, mean=0, std=0.03):
    perturbation_dict = {}
    errors = []
    print(f'Relative errors after perturbations from Normal with mean={mean}, std={std}: ' + ("=" * 30))
    for key in perturbation_keys:
        old_value = input_dict[key]
        perturbation = np.random.normal(loc=mean, scale=std)
        new_value = old_value + (old_value * perturbation)

        epsilon = 1e-6
        error = np.abs(np.abs(old_value - new_value) / (old_value + epsilon))
        errors.append(error)
        print(f'{key}: {error}')

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help="Input file")
    parser.add_argument("-t", "--temp_file", help="Input file")
    parser.add_argument("-e", "--error_file", help="Input file")
    args = parser.parse_args()

    perturb_inputs(args.input_file, args.temp_file, args.error_file)
