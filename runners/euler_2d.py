import hydra
from omegaconf import OmegaConf
import sys
import os
import subprocess
import time
import json
import shutil
from pathlib import Path

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

    if cfg.verbose:
        print(f"\nSimulation completed successfully!")
        print(f"Runtime: {elapsed_time:.2f} seconds")
        print(f"Cost: {cost}")
        print(f"Output: {final_output_dir}")


if __name__ == "__main__":
    main()
