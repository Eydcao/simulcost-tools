#!/usr/bin/env python3
"""
EPOCH Environment Setup Script for CostSci-Tools

This script automatically sets up EPOCH as a git submodule in solvers/epoch/
and compiles the required binaries for different particle-weighting orders.

Prerequisites:
- GNU Fortran compiler (gfortran)
- OpenMPI (openmpi-bin libopenmpi-dev)

Usage:
    python setup_epoch.py

Directory structure after setup:
    solvers/epoch/                    # Git submodule
    solvers/epoch/epoch1d/            # EPOCH 1D source
    solvers/epoch/epoch_bin/          # Compiled binaries
    solvers/epoch/epoch_bin/2nd       # 2nd order binary
    solvers/epoch/epoch_bin/3rd       # 3rd order binary  
    solvers/epoch/epoch_bin/5th       # 5th order binary
"""

import os
import subprocess
import sys
import shutil
import re
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run shell command with error handling"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=check, 
                          capture_output=True, text=True)
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
    
    # Check git
    try:
        run_command("git --version")
        print("✅ Git found")
    except subprocess.CalledProcessError:
        print("❌ Git not found")
        print("Please install git")
        return False
    
    return True


def setup_epoch_submodule():
    """Clone EPOCH as git submodule in solvers/epoch/"""
    print("\n=== Setting up EPOCH git submodule ===")
    
    repo_root = Path(__file__).parent.parent
    solvers_dir = repo_root / "solvers"
    epoch_dir = solvers_dir / "epoch"
    
    # Check if submodule already exists
    if epoch_dir.exists():
        print(f"EPOCH directory already exists at {epoch_dir}")
        print("Updating existing submodule...")
        try:
            run_command("git submodule update --init --recursive", cwd=repo_root)
            return epoch_dir
        except subprocess.CalledProcessError:
            print("Failed to update submodule. Removing and re-adding...")
            shutil.rmtree(epoch_dir)
    
    # Add EPOCH as git submodule using relative path
    try:
        run_command("git submodule add --force https://github.com/Warwick-Plasma/epoch.git solvers/epoch", 
                   cwd=repo_root)
        run_command("git submodule update --init --recursive", cwd=repo_root)
        print(f"✅ EPOCH cloned as submodule in {epoch_dir}")
        return epoch_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone EPOCH submodule: {e}")
        sys.exit(1)


def modify_makefile_for_order(makefile_path, particle_order):
    """Modify Makefile for specific particle order"""
    print(f"Configuring Makefile for {particle_order} order...")
    
    with open(makefile_path, 'r') as f:
        content = f.read()
    
    # Reset all particle shape defines (comment them out)
    content = re.sub(r'^(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_TOPHAT.*)', 
                     r'#\1', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_BSPLINE3.*)', 
                     r'#\1', content, flags=re.MULTILINE)
    
    # Enable specific particle order
    if particle_order == "2nd":
        # Uncomment PARTICLE_SHAPE_TOPHAT for 2nd order
        content = re.sub(r'^#(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_TOPHAT.*)', 
                        r'\1', content, flags=re.MULTILINE)
    elif particle_order == "3rd":
        # Default is 3rd order, no changes needed
        pass
    elif particle_order == "5th":
        # Uncomment PARTICLE_SHAPE_BSPLINE3 for 5th order
        content = re.sub(r'^#(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_BSPLINE3.*)', 
                        r'\1', content, flags=re.MULTILINE)
    
    with open(makefile_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Makefile configured for {particle_order} order")


def compile_epoch_binary(epoch1d_dir, particle_order, output_dir):
    """Compile EPOCH binary for specific particle order"""
    print(f"\n=== Compiling EPOCH for {particle_order} order ===")
    
    makefile_path = epoch1d_dir / "Makefile"
    
    # Clean previous build
    try:
        run_command("make clean", cwd=epoch1d_dir)
    except subprocess.CalledProcessError:
        pass  # Clean might fail if no previous build
    
    # Modify Makefile for this order
    modify_makefile_for_order(makefile_path, particle_order)
    
    # Compile
    try:
        run_command("make COMPILER=gfortran", cwd=epoch1d_dir)
        
        # Copy binary to output location
        binary_src = epoch1d_dir / "bin" / "epoch1d"
        binary_dst = output_dir / particle_order
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy binary
        shutil.copy2(binary_src, binary_dst)
        
        # Make executable
        os.chmod(binary_dst, 0o755)
        
        print(f"✅ {particle_order} order binary compiled and saved to {binary_dst}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to compile {particle_order} order binary: {e}")
        return False


def verify_binaries_different(bin_dir):
    """Verify that all three binaries are different"""
    print("\n=== Verifying binaries are different ===")
    
    binaries = ["2nd", "3rd", "5th"]
    bin_paths = [bin_dir / binary for binary in binaries]
    
    # Check all binaries exist
    for binary_path in bin_paths:
        if not binary_path.exists():
            print(f"❌ Binary not found: {binary_path}")
            return False
    
    # Compare binaries pairwise
    success = True
    for i in range(len(bin_paths)):
        for j in range(i + 1, len(bin_paths)):
            try:
                result = run_command(f"cmp {bin_paths[i]} {bin_paths[j]}", check=False)
                if result.returncode == 0:
                    print(f"❌ {binaries[i]} and {binaries[j]} binaries are identical!")
                    success = False
                else:
                    print(f"✅ {binaries[i]} and {binaries[j]} binaries are different")
            except Exception as e:
                print(f"❌ Failed to compare {binaries[i]} and {binaries[j]}: {e}")
                success = False
    
    return success


def update_runner_paths(repo_root, epoch_bin_dir):
    """Update runner/epoch.py to use new binary paths"""
    print("\n=== Updating runner paths ===")
    
    runner_path = repo_root / "runners" / "epoch.py"
    
    with open(runner_path, 'r') as f:
        content = f.read()
    
    # Update binary paths
    new_2nd_path = str(epoch_bin_dir / "2nd")
    new_3rd_path = str(epoch_bin_dir / "3rd") 
    new_5th_path = str(epoch_bin_dir / "5th")
    
    content = re.sub(r'path_epoch2ndOrder\s*=\s*["\'][^"\']*["\']',
                     f'path_epoch2ndOrder="{new_2nd_path}"', content)
    content = re.sub(r'path_epoch3rdOrder\s*=\s*["\'][^"\']*["\']',
                     f'path_epoch3rdOrder="{new_3rd_path}"', content)
    content = re.sub(r'path_epoch5thOrder\s*=\s*["\'][^"\']*["\']',
                     f'path_epoch5thOrder="{new_5th_path}"', content)
    
    with open(runner_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated runner paths in {runner_path}")


def update_input_deck_physics_table_path(repo_root, epoch_dir):
    """Update input.deck physics table path"""
    print("\n=== Updating input.deck physics table path ===")
    
    input_deck_path = repo_root / "runners" / "input.deck"
    physics_table_path = epoch_dir / "epoch1d" / "src" / "physics_packages" / "TABLES"
    
    with open(input_deck_path, 'r') as f:
        content = f.read()
    
    # Update physics table location
    content = re.sub(r'physics_table_location\s*=\s*[^\n]*',
                     f'physics_table_location = {physics_table_path}/', content)
    
    with open(input_deck_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated physics table path in {input_deck_path}")


def main():
    """Main setup function"""
    print("🚀 EPOCH Environment Setup for CostSci-Tools")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them and run again.")
        sys.exit(1)
    
    repo_root = Path(__file__).parent.parent
    
    # Setup EPOCH submodule
    epoch_dir = setup_epoch_submodule()
    epoch1d_dir = epoch_dir / "epoch1d"
    epoch_bin_dir = epoch_dir / "epoch_bin"
    
    # Check if epoch1d directory exists
    if not epoch1d_dir.exists():
        print(f"❌ EPOCH 1D directory not found at {epoch1d_dir}")
        sys.exit(1)
    
    # Compile binaries for different orders
    orders = ["2nd", "3rd", "5th"]
    success_count = 0
    
    for order in orders:
        if compile_epoch_binary(epoch1d_dir, order, epoch_bin_dir):
            success_count += 1
    
    if success_count != len(orders):
        print(f"\n❌ Only {success_count}/{len(orders)} binaries compiled successfully")
        sys.exit(1)
    
    # Verify binaries are different
    if not verify_binaries_different(epoch_bin_dir):
        print("\n❌ Binary verification failed")
        sys.exit(1)
    
    # Update runner paths
    update_runner_paths(repo_root, epoch_bin_dir)
    
    # Update input.deck physics table path
    update_input_deck_physics_table_path(repo_root, epoch_dir)
    
    print("\n" + "=" * 50)
    print("🎉 EPOCH setup completed successfully!")
    print("\nSetup summary:")
    print(f"  - EPOCH submodule: {epoch_dir}")
    print(f"  - Binaries location: {epoch_bin_dir}")
    print(f"    • 2nd order: {epoch_bin_dir}/2nd")
    print(f"    • 3rd order: {epoch_bin_dir}/3rd") 
    print(f"    • 5th order: {epoch_bin_dir}/5th")
    print(f"  - Updated runner: runners/epoch.py")
    print(f"  - Updated input deck: runners/input.deck")
    print("\nEPOCH is now ready to use with CostSci-Tools!")


if __name__ == "__main__":
    main()