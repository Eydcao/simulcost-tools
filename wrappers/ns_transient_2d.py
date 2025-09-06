import os
import subprocess
import h5py
import numpy as np
import json
from scipy.interpolate import RegularGridInterpolator

def run_sim_ns_transient_2d(profile, boundary_condition, resolution, reynolds_num, cfl, advection_scheme, 
                            vorticity_confinement, relaxation_factor, residual_threshold, total_runtime, 
                            no_dye, cpu, visualization, other_params=None):
    """Run the ns_transient_2d simulation with the given parameters."""
    vor = None if vorticity_confinement == 0.0 else vorticity_confinement
    dir_path = f"sim_res/ns_transient_2d/{profile}_bc{boundary_condition}_res{resolution}_re{reynolds_num}_cfl{cfl}_scheme{advection_scheme}_vor{vor}_relax{relaxation_factor}_residual{residual_threshold}_runtime{total_runtime}_no_dye{no_dye}_cpu{cpu}_vis{visualization}/"
    meta_file_path = os.path.join(dir_path, "meta.json")

    # Check if the directory and meta.json file with the key of cost and num_steps exist
    if os.path.exists(meta_file_path):
        with open(meta_file_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta and "num_steps" in meta:
                return meta["cost"], meta["num_steps"]

    # Build command with parameters
    cmd = f"python runners/ns_transient_2d.py --config-name={profile} boundary_condition={boundary_condition} resolution={resolution} reynolds_num={reynolds_num} cfl={cfl} advection_scheme={advection_scheme} vorticity_confinement={vorticity_confinement} relaxation_factor={relaxation_factor} residual_threshold={residual_threshold} total_runtime={total_runtime} no_dye={no_dye} cpu={cpu} visualization={visualization}"
    
    # Add other parameters if provided
    if other_params:
        for key, value in other_params.items():
            cmd += f" {key}={value}"
    print(f"Running simulation with command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Check if simulation failed
    if result.returncode != 0:
        print(f"Simulation failed with return code {result.returncode}")
        print(f"Error output: {result.stderr}")
        # Return a high cost to indicate failure
        return float('inf'), 0

    # Check if meta.json file was created after simulation
    if not os.path.exists(meta_file_path):
        print(f"Warning: meta.json not found at {meta_file_path} after simulation")
        # Return a high cost to indicate failure
        return float('inf'), 0

    # Load the cost from the meta.json file
    try:
        with open(meta_file_path, "r") as f:
            meta = json.load(f)
        
        if "cost" not in meta or "num_steps" not in meta:
            print(f"Warning: meta.json missing required keys at {meta_file_path}")
            return float('inf'), 0
            
        return meta["cost"], meta["num_steps"]
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading meta.json at {meta_file_path}: {e}")
        return float('inf'), 0


def get_res_ns_transient_2d(profile, boundary_condition, resolution, reynolds_num, cfl, advection_scheme, 
                            vorticity_confinement, relaxation_factor, residual_threshold, total_runtime, 
                            no_dye, cpu, visualization, other_params=None):
    """Load final velocity and pressure fields for given parameters."""
    vor = None if vorticity_confinement == 0.0 else vorticity_confinement
    dir_path = f"sim_res/ns_transient_2d/{profile}_bc{boundary_condition}_res{resolution}_re{reynolds_num}_cfl{cfl}_scheme{advection_scheme}_vor{vor}_relax{relaxation_factor}_residual{residual_threshold}_runtime{total_runtime}_no_dye{no_dye}_cpu{cpu}_vis{visualization}/"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Look for H5 data file first
    data_path = os.path.join(dir_path, "data", "simulation_data.h5")
    if os.path.exists(data_path):
        try:
            with h5py.File(data_path, "r") as f:
                # Get the last timestep data
                fields = f['fields']
                last_idx = -1
                
                U = np.array(fields['vx'][last_idx])  # x-velocity
                V = np.array(fields['vy'][last_idx])  # y-velocity
                P = np.array(fields['pressure'][last_idx])  # pressure
                
                # Get additional fields if available
                dye = None
                vorticity = None
                if 'dye' in fields:
                    dye = np.array(fields['dye'][last_idx])
                if 'vorticity' in fields:
                    vorticity = np.array(fields['vorticity'][last_idx])
                
                return U, V, P, dye, vorticity
        except Exception as e:
            print(f"Error reading H5 data file {data_path}: {e}")
    
    # Fallback: look for NPZ files
    files = [f for f in os.listdir(dir_path) if f.startswith("step_") and f.endswith(".npz")]
    if not files:
        # Trigger a simulation run if no result files are found
        cost, num_steps = run_sim_ns_transient_2d(profile, boundary_condition, resolution, reynolds_num, cfl, advection_scheme, vorticity_confinement, relaxation_factor, residual_threshold, total_runtime, no_dye, cpu, visualization, other_params)
        if cost == float('inf'):
            # Simulation failed, return None to indicate failure
            return None, None, None, None, None
        
        # Check again for H5 file after simulation
        if os.path.exists(data_path):
            try:
                with h5py.File(data_path, "r") as f:
                    fields = f['fields']
                    last_idx = -1
                    
                    U = np.array(fields['vx'][last_idx])
                    V = np.array(fields['vy'][last_idx])
                    P = np.array(fields['pressure'][last_idx])
                    
                    dye = None
                    vorticity = None
                    if 'dye' in fields:
                        dye = np.array(fields['dye'][last_idx])
                    if 'vorticity' in fields:
                        vorticity = np.array(fields['vorticity'][last_idx])
                    
                    return U, V, P, dye, vorticity
            except Exception as e:
                print(f"Error reading H5 data file {data_path} after simulation: {e}")
        
        # Check for NPZ files after simulation
        files = [f for f in os.listdir(dir_path) if f.startswith("step_") and f.endswith(".npz")]
        if not files:
            print(f"Warning: No result files found in {dir_path} after triggering a simulation run.")
            return None, None, None, None, None

    # Load from NPZ file
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    latest_file = files[-1]

    file_path = os.path.join(dir_path, latest_file)
    try:
        data = np.load(file_path)
        U = data['u'] if 'u' in data else data['vx'] if 'vx' in data else None
        V = data['v'] if 'v' in data else data['vy'] if 'vy' in data else None
        P = data['p'] if 'p' in data else data['pressure'] if 'pressure' in data else None
        dye = data['dye'] if 'dye' in data else None
        vorticity = data['vorticity'] if 'vorticity' in data else None
        
        return U, V, P, dye, vorticity
    except Exception as e:
        print(f"Error reading result file {file_path}: {e}")
        return None, None, None, None, None


def interpolate_field(field_src, src_shape, tgt_shape, axis_offsets=(0, 0)):
    """
    Interpolates field_src from src_shape to tgt_shape, with optional axis offset.

    axis_offsets: (dy, dx) offset for staggered alignment, typically (0, 0.5), etc.
    """
    ny_src, nx_src = src_shape
    ny_tgt, nx_tgt = tgt_shape

    x_src = np.linspace(0, 1, nx_src)
    y_src = np.linspace(0, 1, ny_src)
    x_tgt = np.linspace(0, 1, nx_tgt) + axis_offsets[1] / nx_tgt
    y_tgt = np.linspace(0, 1, ny_tgt) + axis_offsets[0] / ny_tgt

    interp = RegularGridInterpolator((y_src, x_src), field_src, bounds_error=False, fill_value=0.0)

    yy, xx = np.meshgrid(y_tgt, x_tgt, indexing='ij')
    points = np.stack([yy.ravel(), xx.ravel()], axis=-1)

    return interp(points).reshape(ny_tgt, nx_tgt)


def compare_res_ns_transient_2d(
    profile1, boundary_condition1, resolution1, reynolds_num1, cfl1, advection_scheme1, vorticity_confinement1, relaxation_factor1, residual_threshold1, total_runtime1, no_dye1, cpu1, visualization1,
    profile2, boundary_condition2, resolution2, reynolds_num2, cfl2, advection_scheme2, vorticity_confinement2, relaxation_factor2, residual_threshold2, total_runtime2, no_dye2, cpu2, visualization2,
    norm_rmse_tolerance,
    other_params1=None, other_params2=None
):
    """
    Compare two sets of results.
    """
    u1, v1, p1, dye1, vorticity1 = get_res_ns_transient_2d(profile1, boundary_condition1, resolution1, reynolds_num1, cfl1, advection_scheme1, vorticity_confinement1, relaxation_factor1, residual_threshold1, total_runtime1, no_dye1, cpu1, visualization1, other_params1)
    u2, v2, p2, dye2, vorticity2 = get_res_ns_transient_2d(profile2, boundary_condition2, resolution2, reynolds_num2, cfl2, advection_scheme2, vorticity_confinement2, relaxation_factor2, residual_threshold2, total_runtime2, no_dye2, cpu2, visualization2, other_params2)
    
    # Check if either simulation failed
    if u1 is None or u2 is None:
        print("One or both simulations failed, cannot compare results")
        return False, float('inf')

    # Check for NaN or infinite values in velocity fields
    if np.any(np.isnan(u1)) or np.any(np.isinf(u1)) or np.any(np.isnan(v1)) or np.any(np.isinf(v1)):
        print("First simulation contains NaN or infinite values in velocity fields")
        return False, float('inf')
    
    if np.any(np.isnan(u2)) or np.any(np.isinf(u2)) or np.any(np.isnan(v2)) or np.any(np.isinf(v2)):
        print("Second simulation contains NaN or infinite values in velocity fields")
        return False, float('inf')

    # Compute RMSE for velocity and pressure
    u2_interp = interpolate_field(u2, u2.shape, u1.shape, axis_offsets=(0.0, 0.5))
    v2_interp = interpolate_field(v2, v2.shape, v1.shape, axis_offsets=(0.5, 0.0))
    p2_interp = interpolate_field(p2, p2.shape, p1.shape, axis_offsets=(0.0, 0.0))
    
    # Check if interpolation produced NaN values
    if np.any(np.isnan(u2_interp)) or np.any(np.isnan(v2_interp)) or np.any(np.isnan(p2_interp)):
        print("Interpolation produced NaN values")
        return False, float('inf')
    
    rmse_u = np.sqrt(np.mean((u1 - u2_interp) ** 2))
    rmse_v = np.sqrt(np.mean((v1 - v2_interp) ** 2))
    rmse_p = np.sqrt(np.mean((p1 - p2_interp) ** 2))
    
    norm_velocity = np.sqrt(u1**2 + v1**2)
    norm_velocity_2_interp = np.sqrt(u2_interp**2 + v2_interp**2)
    rmse_norm_velocity = np.sqrt(np.mean((norm_velocity - norm_velocity_2_interp) ** 2))
    
    # Check if RMSE calculation produced NaN
    if np.isnan(rmse_norm_velocity) or np.isinf(rmse_norm_velocity):
        print(f"RMSE calculation produced NaN/inf: norm_velocity stats: min={np.nanmin(norm_velocity):.6f}, max={np.nanmax(norm_velocity):.6f}, mean={np.nanmean(norm_velocity):.6f}")
        print(f"norm_velocity_2_interp stats: min={np.nanmin(norm_velocity_2_interp):.6f}, max={np.nanmax(norm_velocity_2_interp):.6f}, mean={np.nanmean(norm_velocity_2_interp):.6f}")
        return False, float('inf')
    
    converged = rmse_norm_velocity < norm_rmse_tolerance
    
    print(f"RMSE of norm velocity: {rmse_norm_velocity:.6f}")

    return converged, rmse_norm_velocity


if __name__ == "__main__":
    # Example usage: compare resolution=200 with resolution=400
    profile = "p1"
    boundary_condition = 1
    resolution1 = 200
    resolution2 = 400
    reynolds_num = 1000.0
    cfl = 0.05
    advection_scheme = "cip"
    vorticity_confinement = 0.0
    relaxation_factor = 1.3
    residual_threshold = 1e-2
    total_runtime = 1.0
    no_dye = False
    cpu = True
    visualization = 0
    
    length = 20.0
    breadth = 1.0
    norm_rmse_tolerance = 0.2

    # Compare results
    is_converged, _, _, _ = compare_res_ns_transient_2d(
        profile, boundary_condition, resolution1, reynolds_num, cfl, advection_scheme, vorticity_confinement, relaxation_factor, residual_threshold, total_runtime, no_dye, cpu, visualization,
        profile, boundary_condition, resolution2, reynolds_num, cfl, advection_scheme, vorticity_confinement, relaxation_factor, residual_threshold, total_runtime, no_dye, cpu, visualization,
        norm_rmse_tolerance
    )
    print(f"Convergence achieved: {is_converged}")
