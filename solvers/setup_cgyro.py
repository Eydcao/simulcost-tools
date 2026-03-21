#!/usr/bin/env python3
"""
CGYRO Environment Setup Script for CostSci-Tools

This script automatically sets up CGYRO as a git submodule in solvers/cgyro/
and compiles the required binaries (provided by GACODE).

Prerequisites:
- GNU Fortran compiler (gfortran)
- OpenMPI (openmpi-bin libopenmpi-dev)
- OpenBLAS (libopenblas-dev)
- FFTw3 (libfftw3-dev)

Usage:
    python setup_cgyro.py

Directory structure after setup:
    solvers/cgyro/                    # Git submodule
"""

import os
import subprocess
import sys
import shutil
import re
from pathlib import Path


def run_command(cmd, cwd=None, check=True, env=None):
    """Run shell command with error handling"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=check, capture_output=True, text=True, env=env)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    return result


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")

    # Check gfortran
    try:
        run_command("gfortran --version")
        print("✅ GNU Fortran compiler found")
    except subprocess.CalledProcessError:
        print("❌ GNU Fortran compiler not found")
        print("Please install: sudo apt update && sudo apt install gfortran")
        return False

    # Check mpirun
    try:
        run_command("mpirun --version")
        print("✅ OpenMPI found")
    except subprocess.CalledProcessError:
        print("❌ OpenMPI not found")
        print("Please install: sudo apt update && sudo apt install openmpi-bin libopenmpi-dev")
        return False
    
    # Check OpenBLAS
    openblas_installed = run_command("apt list --installed | grep libopenblas-dev")
    if len(openblas_installed.split('\n') > 1):
        print("✅ OpenBLAS found")
    else:
        print("❌ OpenBLAS not found")
        print("Please install: sudo apt update && sudo apt install libopenblas-dev")
    
    # Check FFTw3
    fftw3_installed = run_command("apt list --installed | grep libfftw3-dev")
    if len(fftw3_installed.split('\n') > 1):
        print("✅ FFTw3 found")
    else:
        print("❌ FFTw3 not found")
        print("Please install: sudo apt update && sudo apt install libfftw3-dev")
    
    # Check git
    try:
        run_command("git --version")
        print("✅ Git found")
    except subprocess.CalledProcessError:
        print("❌ Git not found")
        print("Please install git")
        return False

    return True

def setup_gacode_submodule():
    """Initialize existing CGYRO git submodule"""
    print("\n=== Initializing CGYRO git submodule ===")

    repo_root = Path(__file__).parent.parent
    solvers_dir = repo_root / "solvers"
    gacode_dir = solvers_dir / "gacode"

    # Initialize and update the existing submodule
    try:
        print("Initializing GACODE submodule...")
        run_command("git submodule update --init --recursive", cwd=repo_root)
        print(f"✅ GACODE submodule initialized at {gacode_dir}")
        return gacode_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to initialize GACODE submodule: {e}")
        print("Make sure you're in a git repository and the submodule is properly configured.")
        sys.exit(1)

def modify_platform_file(platform_file_path):
    # Remove ROOT= line in platform file
    to_remove = "ROOT="
    to_replace = "LMATH = "
    replacement = "LMATH = $(FFTWDIR)/openblas-pthread/libopenblas.a $(FFTWDIR)/libfftw3.a $(FFTWDIR)/libfftw3_omp.a $(FFTWDIR)/libfftw3f.a $(FFTWDIR)/libfftw3f_omp.a\n"

    with open(platform_file_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if to_remove not in line and to_replace not in line:
            modified_lines.append(line)
        elif to_replace in line:
            modified_lines.append(replacement)

    with open(platform_file_path, 'w') as f:
        f.writelines(modified_lines)

def compile_cgyro_binary(gacode_dir):
    """Compile CGYRO binary """
    print(f"\n=== Compiling CGYRO ===")

    cgyro_dir = gacode_dir / "cgyro"
    platform_file_path = cgyro_dir / ".." / "platform" / "build" / "make.inc.MINT_OPENMPI"
    gacode_setup_path = gacode_dir / "shared" / "bin" / "gacode_setup"

    # Clean previous build
    try:
        run_command("make clean", cwd=cgyro_dir)
    except subprocess.CalledProcessError:
        pass  # Clean might fail if no previous build

    # Modify platform file
    modify_platform_file(platform_file_path)

    # Set up environment variables for compilation
    my_env = os.environ.copy() 
    my_env["GACODE_PLATFORM"] = "MINT_OPENMPI"
    my_env["GACODE_ROOT"] = gacode_dir
    my_env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    my_env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"

    # Compile
    try:
        # NOTE: conda env should be deactivated during compilation
        run_command(f". {gacode_setup_path} && make", cwd=cgyro_dir, env=my_env)
        # Make executable
        # os.chmod(binary_dst, 0o755)

        print(f"✅ Binary compiled and saved")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to compile binary: {e}")
        return False

def update_runner_paths(repo_root, cgyro_dir):
    """Update runner/cgyro.py to use new binary paths"""
    print("\n=== Updating runner paths ===")

    runner_path = repo_root / "runners" / "cgyro.py"

    with open(runner_path, "r") as f:
        content = f.read()

    # Update binary paths
    bin_dir = cgyro_dir / "bin"

    with open(runner_path, "w") as f:
        f.write(content)

    print(f"✅ Updated runner paths in {runner_path}")


def main():
    """Main setup function"""
    print("🚀 CGYRO Environment Setup for CostSci-Tools")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them and run again.")
        sys.exit(1)

    repo_root = Path(__file__).parent.parent
    solvers_dir = repo_root / "solvers"
    gacode_dir = solvers_dir / "gacode"

    # Setup CGYRO submodule
    gacode_dir = setup_gacode_submodule()

    # Compile binaries
    successfully_compiled = compile_cgyro_binary(gacode_dir)
    if not successfully_compiled:
        print(f"\n❌ Binary compilation failed")
        sys.exit(1)

    # Update runner paths
    # update_runner_paths(repo_root, epoch1d_dir)

    print("\n" + "=" * 50)
    print("🎉 CGYRO setup completed successfully!")
    print("\nSetup summary:")
    print(f"  - GACODE submodule: {gacode_dir}")
    print(f"  - CGYRO Binaries location: {gacode_dir}/cgyro/bin/")
    print(f"  - Updated runner: runners/cgyro.py")
    print(f"  - Updated input file: runners/input.cgyro")
    print("\nCGYRO is now ready to use with CostSci-Tools!")


if __name__ == "__main__":
    main()
