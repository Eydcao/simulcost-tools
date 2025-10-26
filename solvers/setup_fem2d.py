#!/usr/bin/env python3
"""
FEM2D (FastIPC) Environment Setup Script for CostSci-Tools

This script automatically sets up the FastIPC solver dependencies for FEM2D
by compiling the required C++ shared library.

Prerequisites:
- GNU C++ compiler (g++)
- CPU with AVX, FMA, and AVX2 support

Usage:
    python setup_fem2d.py

Directory structure after setup:
    solvers/fastipc_utils/                # FastIPC utilities submodule
    solvers/fastipc_utils/common/math/wrapper/a.so  # Compiled shared library
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
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    return result


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")

    # Check g++
    try:
        run_command("g++ --version")
        print("✅ GNU C++ compiler found")
    except subprocess.CalledProcessError:
        print("❌ GNU C++ compiler not found")
        print("Please install: sudo apt update && sudo apt install g++")
        return False

    # Check CPU features (AVX, AVX2, FMA)
    try:
        result = run_command("lscpu | grep -E 'avx|avx2|fma'", check=False)
        if result.returncode == 0 and result.stdout:
            print("✅ CPU supports AVX/AVX2/FMA")
        else:
            print("⚠️  Warning: Could not verify CPU features (AVX/AVX2/FMA)")
            print("   The compiled library may not work on this CPU")
    except:
        print("⚠️  Warning: Could not check CPU features")

    # Check git
    try:
        run_command("git --version")
        print("✅ Git found")
    except subprocess.CalledProcessError:
        print("❌ Git not found")
        print("Please install git")
        return False

    return True


def setup_fastipc_submodule():
    """Initialize existing FastIPC git submodule"""
    print("\n=== Initializing FastIPC git submodule ===")

    repo_root = Path(__file__).parent.parent
    solvers_dir = repo_root / "solvers"
    fastipc_dir = solvers_dir / "fastipc_utils"

    # Initialize and update the existing submodule
    try:
        print("Initializing FastIPC submodule...")
        run_command("git submodule update --init --recursive", cwd=repo_root)
        print(f"✅ FastIPC submodule initialized at {fastipc_dir}")
        return fastipc_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to initialize FastIPC submodule: {e}")
        print("Make sure you're in a git repository and the submodule is properly configured.")
        sys.exit(1)


def compile_fastipc_library(fastipc_dir):
    """Compile FastIPC C++ shared library"""
    print("\n=== Compiling FastIPC shared library ===")

    wrapper_dir = fastipc_dir / "common" / "math" / "wrapper"

    # Check if wrapper directory exists
    if not wrapper_dir.exists():
        print(f"❌ Wrapper directory not found at {wrapper_dir}")
        print("Make sure the FastIPC submodule is properly initialized")
        sys.exit(1)

    # Compile command
    compile_cmd = (
        "g++ -shared -fPIC -mavx -mfma -mavx2 "
        "-I. -I./Eigen -I./EVCTCD "
        "-o a.so wrapper.cpp EVCTCD/CTCD.cpp"
    )

    try:
        run_command(compile_cmd, cwd=wrapper_dir)

        # Verify the library was created
        lib_path = wrapper_dir / "a.so"
        if lib_path.exists():
            lib_size = lib_path.stat().st_size
            print(f"✅ Shared library compiled successfully")
            print(f"   Location: {lib_path}")
            print(f"   Size: {lib_size / (1024*1024):.2f} MB")
            return True
        else:
            print(f"❌ Library file not created at {lib_path}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to compile shared library: {e}")
        print("\nPossible issues:")
        print("  - Missing source files (wrapper.cpp, EVCTCD/CTCD.cpp)")
        print("  - CPU doesn't support required features (AVX, AVX2, FMA)")
        print("  - Missing or incompatible compiler")
        return False


def verify_library_loads(fastipc_dir):
    """Verify the compiled library can be loaded"""
    print("\n=== Verifying library can be loaded ===")

    wrapper_dir = fastipc_dir / "common" / "math" / "wrapper"
    lib_path = wrapper_dir / "a.so"

    try:
        import ctypes
        so = ctypes.CDLL(str(lib_path))
        print("✅ Library loads successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load library: {e}")
        print("\nPossible issues:")
        print("  - Library was compiled for incompatible CPU features")
        print("  - Missing dependencies in the library")
        return False


def main():
    """Main setup function"""
    print("🚀 FEM2D (FastIPC) Environment Setup for CostSci-Tools")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them and run again.")
        sys.exit(1)

    repo_root = Path(__file__).parent.parent

    # Setup FastIPC submodule
    fastipc_dir = setup_fastipc_submodule()

    # Compile shared library
    if not compile_fastipc_library(fastipc_dir):
        print("\n❌ Compilation failed")
        sys.exit(1)

    # Verify library loads
    if not verify_library_loads(fastipc_dir):
        print("\n⚠️  Library compiled but cannot be loaded")
        print("   The FEM2D solver may not work correctly")

    print("\n" + "=" * 60)
    print("🎉 FEM2D setup completed successfully!")
    print("\nSetup summary:")
    print(f"  - FastIPC submodule: {fastipc_dir}")
    print(f"  - Compiled library: {fastipc_dir}/common/math/wrapper/a.so")
    print("\nFEM2D is now ready to use with CostSci-Tools!")
    print("\nTest the installation:")
    print("  python runners/fem2d.py --help")


if __name__ == "__main__":
    main()
