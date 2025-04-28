# Dummy Solution for Heat1D

Solve 1D heat conduction problems and find proper simulation parameters: CFL and grid number. This document contains the script for dummy-search of these parameters

## Find Convergent CFL

To find the maximum stable CFL number for a given profile while maintaining accuracy:

```bash
python dummy_heat1d.py --task cfl --initial_cfl 1.0 --initial_n_space 100 --tolerance 1e-4
```

**Parameters:**
- `--initial_cfl`: Starting CFL number (will be halved each iteration)
- `--initial_n_space`: Fixed grid resolution for CFL testing
- `--tolerance`: Maximum allowed difference between successive refinements

## Find Convergent Grid Resolution

To find the coarsest sufficient grid resolution for a given profile:

```bash
python dummy_heat1d.py --task n_space --initial_cfl 1.0 --initial_n_space 10 --tolerance 1e-4
```

**Parameters:**
- `--initial_cfl`: Fixed CFL number to use for grid testing
- `--initial_n_space`: Starting grid resolution (will be doubled each iteration)
- `--tolerance`: Maximum allowed difference between successive refinements

## Perturb a New Profile

To create new test profiles with randomized material properties:

```python
import numpy as np
import yaml

def create_new_profile(profile_name):
    # TODO read from the existing p1

    # Randomly generate material properties
    log_h_min = np.log10(0.1)
    log_h_max = np.log10(100)
    log_h = np.random.uniform(log_h_min, log_h_max)
    h = round(10**log_h, 2)
    L = round(np.random.uniform(0.1, 0.2), 3)
    k = round(np.random.uniform(0.5, 1), 2)
    rho = round(np.random.uniform(1000, 2000))
    cp = round(np.random.uniform(800, 1000))
    T_inf = round(np.random.uniform(4, 20))
    T_init = round(np.random.uniform(21, 30))
    record_dt = round(np.random.uniform(1, 4)) * 100

    # TODO save a new copy p2, p3 etc ...
```

**Property Ranges:**
- Heat transfer coefficient (h): 0.1 to 100 (log-uniform)
- Rod length (L): 0.1 to 0.2 m
- Thermal conductivity (k): 0.5 to 1 W/m-K
- Density (ρ): 1000 to 2000 kg/m³
- Specific heat (cp): 800 to 1000 J/kg-K
- Ambient temp (T_inf): 4 to 20°C
- Initial temp (T_init): 21 to 30°C
- Recording interval (record_dt): 100 to 400s
