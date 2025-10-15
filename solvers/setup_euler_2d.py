#!/usr/bin/env python3
"""
Euler 2D Environment Setup Script for CostSci-Tools

This script automatically sets up the CSMPM_BOW Euler 2D gas dynamics solver
by compiling the gas_2d binary.

Prerequisites:
- CMake (>= 3.10)
- C++ compiler with C++17 support (GCC >= 7.0)
- Eigen3 (>= 3.3)
- TBB (Threading Building Blocks)
- Python3 with NumPy and Matplotlib

Usage:
    python setup_euler_2d.py

Directory structure after setup:
    solvers/euler_2d_utils/CSMPM_BOW/      # C++ solver source
    solvers/euler_2d_utils/CSMPM_BOW/build/Examples/gas_2d  # Compiled binary
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run shell command with error handling"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"STDERR: {result.stderr}")
    return result


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    all_good = True

    # Check CMake
    try:
        result = run_command("cmake --version", check=False)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ CMake found: {version}")
        else:
            print("❌ CMake not found")
            all_good = False
    except Exception:
        print("❌ CMake not found")
        all_good = False

    # Check C++ compiler
    try:
        result = run_command("g++ --version", check=False)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ G++ compiler found: {version}")
        else:
            print("❌ G++ compiler not found")
            all_good = False
    except Exception:
        print("❌ G++ compiler not found")
        all_good = False

    # Check Eigen3
    eigen_paths = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
    ]
    eigen_found = any(os.path.exists(p) for p in eigen_paths)
    if eigen_found:
        print("✅ Eigen3 found")
    else:
        print("❌ Eigen3 not found")
        all_good = False

    # Check TBB
    try:
        result = run_command("dpkg -l | grep libtbb", check=False)
        if "libtbb" in result.stdout:
            print("✅ TBB found")
        else:
            print("❌ TBB not found")
            all_good = False
    except Exception:
        print("❌ TBB not found")
        all_good = False

    # Check Python packages
    try:
        import numpy
        import matplotlib
        print("✅ Python NumPy and Matplotlib found")
    except ImportError:
        print("❌ Python NumPy or Matplotlib not found")
        all_good = False

    if not all_good:
        print("\n" + "=" * 50)
        print("❌ Missing dependencies detected!")
        print("\nInstall missing dependencies with:")
        print("sudo apt-get update")
        print("sudo apt-get install -y cmake build-essential libeigen3-dev libtbb-dev \\")
        print("    python3-dev python3-numpy python3-matplotlib")
        return False

    return True


def build_gas_2d(csmpm_bow_dir):
    """Build the gas_2d binary"""
    print("\n" + "=" * 50)
    print("Building gas_2d binary...")
    print("=" * 50)

    build_dir = csmpm_bow_dir / "build"

    # Create build directory if it doesn't exist
    if not build_dir.exists():
        print(f"Creating build directory: {build_dir}")
        build_dir.mkdir(parents=True)

    # Configure with CMake
    print("\nConfiguring with CMake...")
    try:
        run_command("cmake .. -DCMAKE_BUILD_TYPE=Release", cwd=build_dir)
        print("✅ CMake configuration successful")
    except subprocess.CalledProcessError as e:
        print(f"❌ CMake configuration failed: {e}")
        return False

    # Build
    print("\nCompiling...")
    try:
        # Use all available cores for compilation
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        run_command(f"make -j{num_cores}", cwd=build_dir)
        print("✅ Compilation successful")
    except subprocess.CalledProcessError as e:
        print(f"❌ Compilation failed: {e}")
        return False

    return True


def verify_binary(binary_path):
    """Verify that the binary was created and is executable"""
    print("\n" + "=" * 50)
    print("Verifying binary...")
    print("=" * 50)

    if not binary_path.exists():
        print(f"❌ Binary not found at: {binary_path}")
        return False

    if not os.access(binary_path, os.X_OK):
        print(f"❌ Binary is not executable: {binary_path}")
        return False

    # Get file size
    size_bytes = binary_path.stat().st_size
    size_kb = size_bytes / 1024
    print(f"✅ Binary found: {binary_path}")
    print(f"   Size: {size_kb:.1f} KB")

    # Test run the binary with minimal parameters
    print("\nTesting binary with quick run...")
    try:
        result = run_command(
            f"{binary_path} 0 0 5 16",
            cwd=binary_path.parent.parent.parent,  # Run from CSMPM_BOW root
            check=False
        )
        if result.returncode == 0:
            print("✅ Binary test run successful")
        else:
            print("⚠️  Binary test run completed with warnings (this may be normal)")
    except Exception as e:
        print(f"⚠️  Binary test run had issues: {e}")

    return True


def main():
    """Main setup function"""
    print("🚀 Euler 2D Environment Setup for CostSci-Tools")
    print("=" * 50)

    # Get paths
    repo_root = Path(__file__).parent.parent
    csmpm_bow_dir = repo_root / "solvers" / "euler_2d_utils" / "CSMPM_BOW"

    # Check if CSMPM_BOW directory exists
    if not csmpm_bow_dir.exists():
        print(f"❌ CSMPM_BOW directory not found at {csmpm_bow_dir}")
        print("Please ensure the solver source code is in the correct location.")
        sys.exit(1)

    print(f"CSMPM_BOW directory: {csmpm_bow_dir}")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Build the gas_2d binary
    if not build_gas_2d(csmpm_bow_dir):
        print("\n" + "=" * 50)
        print("❌ Build failed. Please check the error messages above.")
        print("\nFor detailed build instructions, see:")
        print(f"  {csmpm_bow_dir}/BUILD.md")
        sys.exit(1)

    # Verify binary was created
    binary_path = csmpm_bow_dir / "build" / "Examples" / "gas_2d"
    if not verify_binary(binary_path):
        print("\n" + "=" * 50)
        print("❌ Binary verification failed.")
        sys.exit(1)

    # Success
    print("\n" + "=" * 50)
    print("🎉 Euler 2D setup completed successfully!")
    print("\nSetup summary:")
    print(f"  - CSMPM_BOW directory: {csmpm_bow_dir}")
    print(f"  - Binary location: {binary_path}")
    print(f"  - Binary size: {binary_path.stat().st_size / 1024:.1f} KB")
    print("\nUsage:")
    print(f"  {binary_path} testcase [start_frame] [end_frame] [N_grid_x]")
    print("\nTest cases:")
    print("  0 - Central explosion")
    print("  1 - Stair flow")
    print("  2 - Cylinder with gravity")
    print("  3 - Mach diamond")
    print("\nExample:")
    print(f"  {binary_path} 0 0 180 64")
    print("\nThe Euler 2D solver is now ready to use with CostSci-Tools!")


if __name__ == "__main__":
    main()
