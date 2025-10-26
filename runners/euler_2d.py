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

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrappers.euler_2d import _read_vtk_structured


def find_gas_2d_binary():
    """Find the gas_2d binary in the expected location"""
    repo_root = Path(__file__).parent.parent
    binary_path = os.path.join(repo_root, "solvers", "euler_2d_utils", "CSMPM_BOW", "build", "Examples", "gas_2d")
    binary_path = Path(binary_path)

    if not binary_path.exists():
        raise FileNotFoundError(
            f"gas_2d binary not found at {binary_path}\n" "Please run: python solvers/setup_euler_2d.py"
        )

    return binary_path


def generate_visualization_plots(output_dir, start_frame, end_frame):
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

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    nx = metadata["parameters"]["n_grid_x"]
    ny = metadata["parameters"]["n_grid_y"]
    aspect_ratio = ny / nx if nx > 0 else 1.0

    # If frame range not specified, get from metadata (good fallback)
    if start_frame is None:
        start_frame = metadata.get("start_frame", 0)
    if end_frame is None:
        end_frame = metadata.get("end_frame", 20)

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

        # --- MODIFIED: Call _read_vtk_structured directly ---
        try:
            # This now returns a dict of 2D (ny, nx) interior-only arrays
            fields = _read_vtk_structured(vtk_file, nx, ny)
        except Exception as e:
            print(f"  Error reading VTK file {vtk_file}: {e}, skipping...")
            continue

        # --- MODIFIED: Data is already 2D ---
        # Extract fields
        density_2d = fields.get("density", fields.get("rho", None))
        pressure_2d = fields.get("pressure", fields.get("p", None))
        vx_2d = fields.get("velocity_x", fields.get("u", None))
        vy_2d = fields.get("velocity_y", fields.get("v", None))

        if density_2d is None or pressure_2d is None:
            print(f"  Warning: Required fields not found in frame {frame}, skipping...")
            continue

        # Reshaping is no longer needed
        # Set defaults if velocity fields are missing
        if vx_2d is None:
            vx_2d = np.zeros((ny, nx))
        if vy_2d is None:
            vy_2d = np.zeros((ny, nx))
        # --- End of modification ---

        # Create 2x2 plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        extent = [0, 1, 0, aspect_ratio]

        # Density
        im0 = axes[0, 0].imshow(density_2d, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        axes[0, 0].set_title(f"Density (Frame {frame})")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        plt.colorbar(im0, ax=axes[0, 0])

        # Pressure
        im1 = axes[0, 1].imshow(pressure_2d, origin="lower", extent=extent, aspect="auto", cmap="plasma")
        axes[0, 1].set_title(f"Pressure (Frame {frame})")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0, 1])

        # Velocity X
        im2 = axes[1, 0].imshow(vx_2d, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
        axes[1, 0].set_title(f"Velocity X (Frame {frame})")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("y")
        plt.colorbar(im2, ax=axes[1, 0])

        # Velocity Y
        im3 = axes[1, 1].imshow(vy_2d, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
        axes[1, 1].set_title(f"Velocity Y (Frame {frame})")
        axes[1, 1].set_xlabel("x")
        axes[1, 0].set_ylabel("y")
        plt.colorbar(im3, ax=axes[1, 1])

        plt.tight_layout()

        # Save plot
        plot_file = plots_dir / f"frame_{frame:03d}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Generated {end_frame - start_frame + 1} visualization plots in {plots_dir}")


@hydra.main(version_base=None, config_path="../run_configs/euler_2d", config_name="p1")
def main(cfg):
    # Print config (for debugging)
    if cfg.verbose:
        print(OmegaConf.to_yaml(cfg))

    # Find binary
    binary_path = find_gas_2d_binary()

    # Construct output directory path with all tunable parameters
    repo_root = Path(__file__).parent.parent
    profile_name = cfg.dump_dir.replace("sim_res/euler_2d/", "")
    output_dir = (
        repo_root
        / "sim_res"
        / "euler_2d"
        / f"{profile_name}_cfl_{cfg.cfl:.3f}_cgtol_{cfg.cg_tolerance:.1e}_nx_{cfg.n_grid_x}"
    )

    # Ensure parent directory exists
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing directory if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Construct command - all parameters are required
    cmd = f"{binary_path} {cfg.testcase} {cfg.start_frame} {cfg.end_frame} {cfg.n_grid_x} {cfg.record_dt} {cfg.cfl} {cfg.cg_tolerance} {output_dir}"

    if cfg.verbose:
        print(f"Running: {cmd}")
        print(f"Output directory: {output_dir}")

    # Run simulation and measure time
    start_time = time.time()

    # Get the working directory for the C++ binary (doesn't matter where we run it from now)
    csmpm_bow_root = binary_path.parent.parent.parent

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=csmpm_bow_root, check=True, capture_output=not cfg.verbose, text=True
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

    # C++ uses the exact directory path we provided
    if not output_dir.exists():
        raise FileNotFoundError(f"C++ binary did not create output directory at {output_dir}")

    # Find metadata file
    meta_path = output_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"C++ binary did not create metadata file in {output_dir}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    # Add Python-side runtime measurement
    if "runtime_seconds" not in metadata:
        metadata["runtime_seconds"] = elapsed_time
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

    cost = metadata["cost"]
    Nx = metadata["parameters"]["n_grid_x"]
    Ny = metadata["parameters"]["n_grid_y"]

    # Generate visualization plots
    try:
        # --- MODIFIED: Pass start_frame and end_frame from the config ---
        generate_visualization_plots(output_dir, cfg.start_frame, cfg.end_frame)
    except Exception as e:
        print(f"Warning: Failed to generate visualization plots: {e}")

    if cfg.verbose:
        print(f"\nSimulation completed successfully!")
        print(f"Runtime: {elapsed_time:.2f} seconds")
        print(f"Cost: {cost}")
        print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
