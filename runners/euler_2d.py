import hydra
from omegaconf import OmegaConf
import sys
import os
import subprocess
import time
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_gas_2d_binary():
    """Find the gas_2d binary in the expected location"""
    repo_root = Path(__file__).parent.parent
    binary_path = os.path.join(repo_root, "solvers", "euler_2d_utils", "CSMPM_BOW", "build", "Examples", "gas_2d")
    binary_path = Path(binary_path)

    if not binary_path.exists():
        raise FileNotFoundError(
            f"gas_2d binary not found at {binary_path}\n"
            "Please run: python solvers/setup_euler_2d.py"
        )

    return binary_path


def read_vtk_for_viz(vtk_path, nx_interior, ny_interior):
    """Read VTK file and extract interior cells (removing ghost layers).

    Args:
        vtk_path: Path to VTK file
        nx_interior: Interior grid size in x (from meta.json)
        ny_interior: Interior grid size in y (from meta.json)

    Returns:
        tuple: (fields dict, nx, ny) where fields contains density, pressure, velocity_x, velocity_y
               for interior cells only
    """
    # Import from wrappers
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from wrappers.euler_2d import read_vtk_structured, _extract_interior_cells

    # Read VTK with ghost cells
    data = read_vtk_structured(vtk_path)

    # Extract interior cells for each field
    # _extract_interior_cells expects (field_flat, nx_interior, ny_interior)
    fields = {}
    for field_name in ['density', 'pressure', 'velocity_x', 'velocity_y']:
        if field_name in data:
            fields[field_name] = _extract_interior_cells(data[field_name], nx_interior, ny_interior)

    return fields, nx_interior, ny_interior


def generate_visualization_plots(output_dir, start_frame=None, end_frame=None):
    """Generate 2x2 visualization plots for each frame.

    Args:
        output_dir: Path to simulation output directory (contains vtk/ and meta.json)
        start_frame: Starting frame number (if None, read from meta.json)
        end_frame: Ending frame number (if None, read from meta.json)

    This function can be called standalone to generate plots for existing simulations.
    """
    from pathlib import Path

    final_output_dir = Path(output_dir)
    vtk_dir = final_output_dir / "vtk"
    plots_dir = final_output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Read grid dimensions from meta.json
    meta_path = final_output_dir / "meta.json"
    if not meta_path.exists():
        print(f"  Warning: meta.json not found at {meta_path}, skipping visualization...")
        return

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    nx = metadata['n_grid_x']
    ny = metadata['n_grid_y']
    aspect_ratio = ny / nx if nx > 0 else 1.0

    # If frame range not specified, get from metadata
    if start_frame is None:
        start_frame = metadata.get('start_frame', 0)
    if end_frame is None:
        end_frame = metadata.get('end_frame', 20)

    print(f"\nGenerating visualization plots for {output_dir}...")
    print(f"  Grid: {nx}x{ny}, aspect ratio: {aspect_ratio:.6f}")
    print(f"  Frames: {start_frame} to {end_frame}")

    for frame in range(start_frame, end_frame + 1):
        # Try multiple filename patterns (gas_frame_N.vtk or output_N.vtk)
        vtk_file = vtk_dir / f"gas_frame_{frame}.vtk"
        if not vtk_file.exists():
            vtk_file = vtk_dir / f"output_{frame}.vtk"
        if not vtk_file.exists():
            print(f"  Warning: VTK file not found for frame {frame}, skipping...")
            continue

        # Read VTK data and extract interior cells
        try:
            fields, nx_vtk, ny_vtk = read_vtk_for_viz(vtk_file, nx, ny)
        except Exception as e:
            print(f"  Error reading VTK file {vtk_file}: {e}, skipping...")
            continue

        # Extract fields
        density = fields.get('density', fields.get('rho', None))
        pressure = fields.get('pressure', fields.get('p', None))
        vx = fields.get('velocity_x', fields.get('u', None))
        vy = fields.get('velocity_y', fields.get('v', None))

        if density is None or pressure is None:
            print(f"  Warning: Required fields not found in frame {frame}, skipping...")
            continue

        # Reshape to 2D (assuming y-major, x-fastest ordering)
        density_2d = density.reshape((ny_vtk, nx_vtk))
        pressure_2d = pressure.reshape((ny_vtk, nx_vtk))
        vx_2d = vx.reshape((ny_vtk, nx_vtk)) if vx is not None else np.zeros((ny_vtk, nx_vtk))
        vy_2d = vy.reshape((ny_vtk, nx_vtk)) if vy is not None else np.zeros((ny_vtk, nx_vtk))

        # Create 2x2 plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        extent = [0, 1, 0, aspect_ratio]

        # Density
        im0 = axes[0, 0].imshow(density_2d, origin='lower', extent=extent,
                                aspect='auto', cmap='viridis')
        axes[0, 0].set_title(f'Density (Frame {frame})')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, 0])

        # Pressure
        im1 = axes[0, 1].imshow(pressure_2d, origin='lower', extent=extent,
                                aspect='auto', cmap='plasma')
        axes[0, 1].set_title(f'Pressure (Frame {frame})')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 1])

        # Velocity X
        im2 = axes[1, 0].imshow(vx_2d, origin='lower', extent=extent,
                                aspect='auto', cmap='RdBu_r')
        axes[1, 0].set_title(f'Velocity X (Frame {frame})')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1, 0])

        # Velocity Y
        im3 = axes[1, 1].imshow(vy_2d, origin='lower', extent=extent,
                                aspect='auto', cmap='RdBu_r')
        axes[1, 1].set_title(f'Velocity Y (Frame {frame})')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1, 1])

        plt.tight_layout()

        # Save plot
        plot_file = plots_dir / f"frame_{frame:03d}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Generated {end_frame - start_frame + 1} visualization plots in {plots_dir}")



@hydra.main(version_base=None, config_path="../run_configs/euler_2d", config_name="p1")
def main(cfg):
    # Print config (for debugging)
    if cfg.verbose:
        print(OmegaConf.to_yaml(cfg))

    # Find binary
    binary_path = find_gas_2d_binary()

    # Get testcase info from config
    testcase = cfg.testcase
    testcase_name = cfg.testcase_name
    aspect_ratio = cfg.aspect_ratio

    Nx = cfg.n_grid_x
    # Ny will be determined by C++ and read from metadata
    # For now, estimate for finding the output directory
    Ny_estimate = int(round(aspect_ratio * Nx))

    # The binary creates output in CSMPM_BOW root directory
    csmpm_bow_root = binary_path.parent.parent.parent

    # Find the actual output directory created by C++ binary
    # It will be named like: testcase_name/_Nx_Ny/
    # TODO the whole output dir handling is trouble some, make that a input arg to the c++ binary and avoid all this searching
    # output directly into the correct location
    testcase_output_dir = csmpm_bow_root / testcase_name

    # Construct command with all parameters
    cmd = f"{binary_path} {testcase} {cfg.start_frame} {cfg.end_frame} {Nx}"

    # Add optional parameters if specified
    if hasattr(cfg, 'record_dt') and cfg.record_dt is not None:
        cmd += f" {cfg.record_dt}"

        # CFL and cg_tolerance can only be passed if record_dt is passed
        if hasattr(cfg, 'cfl') and cfg.cfl is not None:
            cmd += f" {cfg.cfl}"

            if hasattr(cfg, 'cg_tolerance') and cfg.cg_tolerance is not None:
                cmd += f" {cfg.cg_tolerance}"

    if cfg.verbose:
        print(f"Running: {cmd}")
        print(f"Working directory: {csmpm_bow_root}")

    # Run simulation and measure time
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=csmpm_bow_root,
            check=True,
            capture_output=not cfg.verbose,
            text=True
        )

        if cfg.verbose and result.stdout:
            print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Error running simulation: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        raise

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Find the actual output directory created by C++ (Ny may differ from estimate)
    binary_output_dir = None
    Ny = Ny_estimate  # Default, will be overridden if found

    if testcase_output_dir.exists():
        # Look for directory matching _Nx_*
        for subdir in testcase_output_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith(f"_{Nx}_"):
                binary_output_dir = subdir
                # Extract actual Ny from directory name
                Ny = int(subdir.name.split('_')[2])
                break

    if binary_output_dir is None:
        print(f"Error: Could not find output directory for testcase {testcase}, Nx={Nx}")
        if testcase_output_dir.exists():
            print(f"Available directories in {testcase_output_dir}:")
            for subdir in testcase_output_dir.iterdir():
                print(f"  {subdir.name}")
        raise FileNotFoundError(f"Expected output directory not found in {testcase_output_dir}")

    # Now we have Ny, construct final output location with all tunable parameters
    # Format: {profile}_cfl_{cfl}_cgtol_{cg_tol}_nx_{nx}
    repo_root = Path(__file__).parent.parent
    cfl_val = cfg.cfl if hasattr(cfg, 'cfl') and cfg.cfl is not None else 0.5
    cgtol_val = cfg.cg_tolerance if hasattr(cfg, 'cg_tolerance') and cfg.cg_tolerance is not None else 1.0e-7
    final_output_dir = repo_root / f"{cfg.dump_dir.replace('sim_res/euler_2d/', '')}_cfl_{cfl_val:.3f}_cgtol_{cgtol_val:.1e}_nx_{Nx}"
    final_output_dir = repo_root / "sim_res" / "euler_2d" / final_output_dir.name

    # Move output to final location
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_output_dir.exists():
        shutil.rmtree(final_output_dir)
    shutil.move(str(binary_output_dir), str(final_output_dir))

    if cfg.verbose:
        print(f"Moved output from {binary_output_dir} to {final_output_dir}")

    num_cells = Nx * Ny
    num_frames = cfg.end_frame - cfg.start_frame + 1

    # Load or create metadata
    # NOTE: Ideally, the C++ binary should output all metadata fields including:
    # cost, n_grid_x, n_grid_y, testcase, start_frame, end_frame, cfl, cg_tolerance, record_dt
    # This would eliminate the need for Python-side fallback logic and complex if/else branching.
    meta_path = final_output_dir / "meta.json"

    # Load existing metadata from C++ binary if available
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Extract grid dimensions from C++ output if available
        if "parameters" in metadata and "n_grid_x" in metadata["parameters"]:
            Nx = int(metadata["parameters"]["n_grid_x"])
            Ny = int(metadata["parameters"]["n_grid_y"])
    else:
        print(f"Warning: meta.json not found in output directory. Using fallback metadata creation.")
        metadata = {}

    # Add/update all required metadata fields
    metadata["cost"] = metadata.get("cost", num_cells * num_frames)
    metadata["runtime_seconds"] = elapsed_time
    metadata["testcase"] = int(testcase)
    metadata["testcase_name"] = testcase_name
    metadata["n_grid_x"] = int(Nx)
    metadata["n_grid_y"] = int(Ny)
    metadata["start_frame"] = int(cfg.start_frame)
    metadata["end_frame"] = int(cfg.end_frame)
    metadata["num_frames"] = int(num_frames)
    metadata["num_cells"] = int(num_cells)

    # Add optional parameters
    if hasattr(cfg, 'record_dt') and cfg.record_dt is not None:
        metadata["record_dt"] = float(cfg.record_dt)
    if hasattr(cfg, 'cfl') and cfg.cfl is not None:
        metadata["cfl"] = float(cfg.cfl)
    if hasattr(cfg, 'cg_tolerance') and cfg.cg_tolerance is not None:
        metadata["cg_tolerance"] = float(cfg.cg_tolerance)

    # Write metadata
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    cost = metadata["cost"]

    # Generate visualization plots
    try:
        generate_visualization_plots(final_output_dir)
    except Exception as e:
        print(f"Warning: Failed to generate visualization plots: {e}")

    if cfg.verbose:
        print(f"\nSimulation completed successfully!")
        print(f"Runtime: {elapsed_time:.2f} seconds")
        print(f"Cost: {cost}")
        print(f"Output: {final_output_dir}")


if __name__ == "__main__":
    main()
