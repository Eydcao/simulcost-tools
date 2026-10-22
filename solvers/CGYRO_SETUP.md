# CGYRO Setup Guide for CostSci-Tools

This guide provides instructions for setting up CGYRO for use with the CostSci-Tools parameter optimization framework.

## Overview

CGYRO is a global-spectral gyrokinetic code. The CostSci-Tools integration requires:

- **Linux environment** (required for CGYRO compilation)
- **CGYRO binaries** compiled from GACODE suite
- **Proper directory organization** within the solvers/ folder

## Automated Setup (Recommended)

### Prerequisites

Install the required dependencies:

```bash
sudo apt update
sudo apt install gfortran openmpi-bin libopenmpi-dev libopenblas-dev libfftw3-dev
```

### Run Setup Script

Execute the automated setup script from the repository root:

```bash
cd /path/to/costsci-tools
python solvers/setup_cgyro.py
```

The script will automatically:

1. Initialize the existing GACODE git submodule in `solvers/gacode/`
2. Configure platform files for compiling CGYRO using the GACODE suite on a Linux system
3. Compile CGYRO binaries

### Directory Structure After Setup

```
solvers/gacode/                   # Git submodule
├── cgyro/                        # CGYRO source code
│   ├── bin/                      # Binary directory
│   │   ├── cgyro                 # Latest compiled binary
│   └── Makefile                  # Used for compilation
```

## Manual Setup (For Reference)

If you need to set up CGYRO manually or troubleshoot the automated script:

### 1. Initialize GACODE Submodule

Since GACODE is already included as a git submodule in this repository:

```bash
git submodule update --init --recursive
cd solvers/gacode/
```

### 2. Configure platform files for use with Linux system

```bash
vi platform/build/make.inc.MINT_OPENMPI"
```

Once you've opened the platform file in Vim, carry out the following:
1. Remove the following line:
```bash
ROOT=/home/candy/GIT
```
2. Replace the following line:
```bash
LMATH = ${ROOT}/OpenBLAS/libopenblas.a $(FFTWDIR)/libfftw3.a $(FFTWDIR)/libfftw3_omp.a $(FFTWDIR)/libfftw3f.a $(FFTWDIR)/libfftw3f_omp.a
```
With the following line:
```bash
LMATH = $(FFTWDIR)/openblas-pthread/libopenblas.a $(FFTWDIR)/libfftw3.a $(FFTWDIR)/libfftw3_omp.a $(FFTWDIR)/libfftw3f.a $(FFTWDIR)/libfftw3f_omp.a\n
```

### 3. Compile CGYRO

Before compiling using the commands below, ensure that there is no active conda or virtual environment. 

```bash
cd cgyro/
make clean
export GACODE_PLATFORM=MINT_OPENMPI
export GACODE_ROOT=complete/path/to/solvers/gacode
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
. ${GACODE_ROOT}/shared/bin/gacode_setup && make
```

## Usage

Once set up, CGYRO simulations will automatically use the compiled CGYRO binaries with 8 parallel threads.

## Troubleshooting

### Compilation Errors

- Ensure gfortran, OpenMPI, OpenBLAS, and FFTw3 are properly installed
- Check that you're on a Linux system (CGYRO is stable on Linux)
- Verify git submodule was initialized correctly

### Binary Issues

- Re-run the setup script or follow the manual installation instructions
- Ensure `make clean` was run between compilations

### Runtime Errors

- Ensure binaries have execute permissions (`chmod +x`)

## Parameter Optimization

CGYRO supports optimization of 7 parameters:

- **`n_radial`**: Controls the number of radial wavenumbers (radial Fourier harmonics) to retain in simulation [4→]
- **`n_theta`**: Controls the number of poloidal gridpoints [6→]
- **`n_xi`**: Controls the number of Legendre pseudospectral meshpoints to retain in simulation [6→]
- **`n_energy`**: Controls the number of generalized-Laguerre pseudospectral meshpoints to retain in simulation [4→]
- **`freq_tol`**: Controls the eigenvalue convergence tolerance for linear simulations [1e-5→]
- **`error_tol`**: Controls the error tolerance for adaptive time-stepping [1e-6→]
- **`delta_t`**: Controls the initial simulation timestep, which is adaptively modified during runtime [1e-2→]

The automated dummy solutions will explore these parameter spaces across 3 precision levels and 12 physics profiles with plasma parameters similar to DIII-D cases.
