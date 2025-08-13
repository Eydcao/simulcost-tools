import os
import subprocess
import h5py
import numpy as np
import json
from scipy.interpolate import RegularGridInterpolator


def run_sim_ns_channel_2d(profile, boundary_type, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold):
    """Run the ns_channel_2d simulation with the given parameters."""
    dir_path = f"sim_res/ns_channel_2d/{profile}_{boundary_type}_mesh_{mesh_x}_{mesh_y}_relax_{omega_u}_{omega_v}_{omega_p}_error_{diff_u_threshold}_{diff_v_threshold}_itererror_{res_iter_v_threshold}/"
    meta_file_path = os.path.join(dir_path, "meta.json")

    # Check if the directory and meta.json file with the key of cost and num_steps exist
    if os.path.exists(meta_file_path):
        with open(meta_file_path, "r") as f:
            meta = json.load(f)
            if "cost" in meta and "num_steps" in meta:
                return meta["cost"], meta["num_steps"]

    # Run the simulation if the directory or meta.json file does not exist
    cmd = f"python costsci_tools/runners/ns_channel_2d.py --config-name={profile} mesh_x={mesh_x} mesh_y={mesh_y} omega_u={omega_u} omega_v={omega_v} omega_p={omega_p} diff_u_threshold={diff_u_threshold} diff_v_threshold={diff_v_threshold} res_iter_v_threshold={res_iter_v_threshold} boundary_condition={boundary_type}"
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


def get_res_ns_channel_2d(profile, boundary_type, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold):
    """Load final velocity and pressure fields for given parameters."""
    dir_path = f"sim_res/ns_channel_2d/{profile}_{boundary_type}_mesh_{mesh_x}_{mesh_y}_relax_{omega_u}_{omega_v}_{omega_p}_error_{diff_u_threshold}_{diff_v_threshold}_itererror_{res_iter_v_threshold}/"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # Find the latest result file in the directory
    files = [f for f in os.listdir(dir_path) if f.startswith("res_") and f.endswith(".h5")]
    if not files:
        # Trigger a simulation run if no result files are found
        cost, num_steps = run_sim_ns_channel_2d(profile, boundary_type, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold)
        if cost == float('inf'):
            # Simulation failed, return None to indicate failure
            return None, None, None
        files = [f for f in os.listdir(dir_path) if f.startswith("res_") and f.endswith(".h5")]
        if not files:
            print(f"Warning: No result files found in {dir_path} after triggering a simulation run.")
            return None, None, None

    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    latest_file = files[-1]

    file_path = os.path.join(dir_path, latest_file)
    try:
        with h5py.File(file_path, "r") as f:
            U = np.array(f["u"])
            V = np.array(f["v"])
            P = np.array(f["p"])
        return U, V, P
    except Exception as e:
        print(f"Error reading result file {file_path}: {e}")
        return None, None, None

def compute_metrics(u, v, p, length, breadth, mass_tolerance):
    """Compute physical metrics for NS solution."""

    # Mass (mean value)
    source = np.zeros_like(p)
    my, mx = p.shape
    my = my - 2  # Exclude ghost cells
    mx = mx - 2  # Exclude ghost cells
    dx = length / mx
    dy = breadth / my
    source[1:my+1, 1:mx+1] = dy * (u[1:my+1, 1:mx+1] - u[1:my+1, 0:mx]) + dx * (v[1:my+1, 1:mx+1] - v[0:my, 1:mx+1])
    mass = np.sum(source**2)
    mass_conserved = mass < mass_tolerance

    return {
        "mass_conserved": mass_conserved,
    }

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

    interp = RegularGridInterpolator((y_src, x_src), field_src, bounds_error=False, fill_value=None)

    yy, xx = np.meshgrid(y_tgt, x_tgt, indexing='ij')
    points = np.stack([yy.ravel(), xx.ravel()], axis=-1)

    return interp(points).reshape(ny_tgt, nx_tgt)


def compare_res_ns_channel_2d(
    profile1, boundary_type1, mesh_x1, mesh_y1, omega_u1, omega_v1, omega_p1, diff_u_threshold1, diff_v_threshold1, res_iter_v_threshold1,
    profile2, boundary_type2, mesh_x2, mesh_y2, omega_u2, omega_v2, omega_p2, diff_u_threshold2, diff_v_threshold2, res_iter_v_threshold2,
    length, breadth,
    mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
):
    """
    Compare two sets of results.
    """
    u1, v1, p1 = get_res_ns_channel_2d(profile1, boundary_type1, mesh_x1, mesh_y1, omega_u1, omega_v1, omega_p1, diff_u_threshold1, diff_v_threshold1, res_iter_v_threshold1)
    u2, v2, p2 = get_res_ns_channel_2d(profile2, boundary_type2, mesh_x2, mesh_y2, omega_u2, omega_v2, omega_p2, diff_u_threshold2, diff_v_threshold2, res_iter_v_threshold2)
    
    # Check if either simulation failed
    if u1 is None or u2 is None:
        print("One or both simulations failed, cannot compare results")
        return False, float('inf'), float('inf'), float('inf'), False, False

    # Compute RMSE for velocity and pressure
    u2_interp = interpolate_field(u2, u2.shape, u1.shape, axis_offsets=(0.0, 0.5))
    v2_interp = interpolate_field(v2, v2.shape, v1.shape, axis_offsets=(0.5, 0.0))
    p2_interp = interpolate_field(p2, p2.shape, p1.shape, axis_offsets=(0.0, 0.0))
    
    rmse_u = np.sqrt(np.mean((u1 - u2_interp) ** 2))
    rmse_v = np.sqrt(np.mean((v1 - v2_interp) ** 2))
    rmse_p = np.sqrt(np.mean((p1 - p2_interp) ** 2))
    
    # Mass conservation check
    mass_conserved1 = compute_metrics(u1, v1, p1, length, breadth, mass_tolerance)["mass_conserved"]
    mass_conserved2 = compute_metrics(u2, v2, p2, length, breadth, mass_tolerance)["mass_conserved"]
    
    converged = (
        mass_conserved1 and mass_conserved2 and
        rmse_u < u_rmse_tolerance and
        rmse_v < v_rmse_tolerance and
        rmse_p < p_rmse_tolerance
    )
    
    print(f"RMSE of u: {rmse_u:.6f}, RMSE of v: {rmse_v:.6f}, RMSE of p: {rmse_p:.6f}")
    print(f"Mass conservation for profile 1: {mass_conserved1}, profile 2: {mass_conserved2}")

    return converged, rmse_u, rmse_v, rmse_p, mass_conserved1, mass_conserved2


if  __name__ == "__main__":
    # Example usage: compare mesh_x=100, mesh_y=50 with mesh_x=200, mesh_y=50
    profile = "p7"
    boundary_type = "back_stair_flow"
    mesh_x1 = 100
    mesh_y1 = 50
    mesh_x2 = 200
    mesh_y2 = 50
    omega_u = 0.5
    omega_v = 0.5
    omega_p = 0.5
    diff_u_threshold = 1e-6
    diff_v_threshold = 1e-6
    
    length = 20.0
    breadth = 1.0
    mass_tolerance = 1e-8
    u_rmse_tolerance = 1e-3
    v_rmse_tolerance = 1e-3
    p_rmse_tolerance = 1e-3
    res_iter_v_threshold = 1e-4

    # Compare results
    is_converged, _, _, _, _, _ = compare_res_ns_channel_2d(
        profile, boundary_type, mesh_x1, mesh_y1, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
        profile, boundary_type, mesh_x2, mesh_y2, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
        length, breadth,
        mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
    )
    print(f"Convergence achieved: {is_converged}")