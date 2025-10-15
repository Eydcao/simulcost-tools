# Euler 2D Setup Guide for CostSci-Tools

This guide provides instructions for setting up the CSMPM_BOW Euler 2D gas dynamics solver for use with the CostSci-Tools parameter optimization framework.

## Overview

CSMPM_BOW is a 2D Compressible Euler gas dynamics solver implemented in C++. The CostSci-Tools integration requires:

- **Linux environment** (recommended for compilation)
- **Compiled gas_2d binary** from the CSMPM_BOW codebase
- **Proper directory organization** within the solvers/ folder

## Automated Setup (Recommended)

### Prerequisites

Install the required dependencies:

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libeigen3-dev libtbb-dev \
    python3-dev python3-numpy python3-matplotlib
```

### Run Setup Script

Execute the automated setup script from the repository root:

```bash
cd /path/to/costsci-tools
python solvers/setup_euler_2d.py
```

The script will automatically:

1. Check for required dependencies (CMake, Eigen3, TBB, Python)
2. Navigate to the CSMPM_BOW directory
3. Create build directory and configure with CMake
4. Compile the gas_2d binary
5. Verify the binary was created successfully
6. Print the binary location for wrapper use

### Directory Structure After Setup

```
solvers/euler_2d_utils/
└── CSMPM_BOW/                    # Standalone C++ solver
    ├── build/                    # Build directory
    │   └── Examples/
    │       └── gas_2d            # Compiled binary (~580 KB)
    ├── Examples/
    │   └── gas_2d.cpp           # 2D gas simulation source
    ├── Libs/                    # Core libraries
    │   ├── BowReplacement/      # BOW framework replacements
    │   ├── EOS/                 # Equation of state
    │   ├── EulerGas/            # Euler gas operators
    │   ├── RPSolver/            # WENO reconstruction
    │   ├── TimeIntegration/     # TVD Runge-Kutta
    │   ├── LinearProjectionSys/ # Projection methods
    │   ├── Simulator/           # Gas simulator
    │   └── IO/                  # VTK/PLY output
    ├── CMakeLists.txt           # Build configuration
    ├── BUILD.md                 # Detailed build instructions
    ├── DEPENDENCIES.md          # Dependency information
    ├── QUICKSTART.md            # Quick start guide
    └── README.md                # Project overview
```

## Manual Setup (For Reference)

If you need to set up the solver manually or troubleshoot the automated script:

### 1. Install Dependencies

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libeigen3-dev libtbb-dev \
    python3-dev python3-numpy python3-matplotlib
```

### 2. Navigate to CSMPM_BOW Directory

```bash
cd solvers/euler_2d_utils/CSMPM_BOW
```

### 3. Build the Project

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake (Release mode for performance)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile
make -j$(nproc)

# Verify binary was created
ls -lh Examples/gas_2d
```

Expected output: `Examples/gas_2d` binary (~580 KB)

### 4. Test the Binary

```bash
# Run test case 0 (central explosion) with small grid for quick test
./Examples/gas_2d 0 0 5 16
```

Expected output:
- Log messages showing simulation progress
- Creates directory `central_boom_2d_16_16/`
- Generates PLY files and optionally VTK files

## Usage

Once set up, the gas_2d binary can be used with different test cases and parameters:

```bash
# Basic syntax
./gas_2d testcase [start_frame] [end_frame] [N_grid_x]

# Test cases:
# 0 - Central explosion (circular high-pressure region)
# 1 - Stair flow (supersonic flow over step geometry)
# 2 - Cylinder with gravity (explosion with gravitational effects)
# 3 - Mach diamond (supersonic jet forming shock diamonds)

# Examples:
./gas_2d 0              # Central explosion, 64x64 grid, frames 0-180
./gas_2d 1 0 100        # Stair flow, frames 0-100
./gas_2d 0 0 180 128    # Central explosion, 128x128 grid
```

### Output Files

The binary creates output in automatically named directories:
- `central_boom_2d_{Nx}_{Ny}/` - PLY point cloud files
- `central_boom_2d_{Nx}_{Ny}/vtk/` - VTK structured grid files (if enabled)

Each output includes:
- **gas_{frame}.ply** - Point cloud with density, pressure, velocity, schlieren
- **gas_density_{frame}.vtk** - VTK files for ParaView visualization

## Solver Features

### Physics
- Compressible Euler equations for ideal gas
- 2D Cartesian grids
- High-order WENO reconstruction (2nd/3rd order)
- TVD Runge-Kutta time integration (2nd/3rd order)
- Riemann solver (Local Lax-Friedrichs)
- Gravity and source terms support

### Numerical Methods
- Grid-based finite volume method
- Adaptive time stepping with CFL condition
- Linear projection for constraint enforcement
- Parallel execution with TBB

### Boundary Conditions
- Inlet (prescribed flow)
- Outlet (extrapolation)
- Wall (no-slip/slip)
- Free boundaries

## Parameter Optimization

The Euler 2D solver supports optimization of several parameters:

- **`n_grid_x`**: Spatial grid resolution in x direction (e.g., 16, 32, 64, 128, 256, 512)
  - `n_grid_y` is computed from `n_grid_x` and aspect ratio (test case dependent)
- **`cfl`**: CFL number for timestep stability (typical range: 0.1-0.9)
  - Controls time step size relative to grid spacing and wave speed
- **`cg_tolerance`**: CG solver convergence tolerance for pressure projection (typical range: 1e-9 to 1e-5)
  - Too tight leads to unnecessary iterations; too loose may cause divergence
- **`record_dt`**: Time between output frames (affects total simulation time)
- **Test case selection**: Different physics scenarios (0-3)

These parameters control the trade-off between accuracy and computational cost.

## Troubleshooting

### Compilation Errors

**Eigen3 Not Found:**
```bash
# Manually specify Eigen3 path
cmake .. -DEigen3_DIR=/usr/share/eigen3/cmake
```

**TBB Not Found:**
```bash
# Ensure TBB is installed
sudo apt-get install libtbb-dev
```

**C++17 Support:**
```bash
# Check compiler version (need GCC >= 7.0)
g++ --version
```

### Runtime Errors

**Binary Not Found:**
```bash
# Verify binary exists
ls -lh solvers/euler_2d_utils/CSMPM_BOW/build/Examples/gas_2d
```

**Output Directory Errors:**
- Ensure write permissions in the working directory
- Check available disk space: `df -h`

**Segmentation Faults:**
- Try smaller grid resolution
- Check memory availability
- Rebuild in Debug mode for more info: `cmake .. -DCMAKE_BUILD_TYPE=Debug`

### Performance Issues

- Use **Release** build for production runs (10-100x faster than Debug)
- Start with small grids (e.g., 32x32) for testing
- Increase grid resolution gradually
- Monitor memory usage (scales with `N_grid_x^2`)

## Integration with CostSci-Tools

The wrapper interface will:

1. **Run simulations** by executing the binary with specified parameters
2. **Parse output** from PLY files (density, pressure, velocity fields)
3. **Calculate cost** based on grid resolution and simulation time
4. **Compare results** between different parameter sets for convergence checking

See `wrappers/euler_2d.py` for the Python interface implementation.

## References

For more detailed information:
- **BUILD.md** - Comprehensive build instructions and troubleshooting
- **DEPENDENCIES.md** - Dependency installation details
- **QUICKSTART.md** - 5-minute getting started guide
- **README.md** - Project overview and architecture

---

**Note**: This solver was originally part of the BOW physics framework and has been refactored as a standalone project for easier integration.
