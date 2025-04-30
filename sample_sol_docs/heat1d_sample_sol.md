# Dummy Solution for Heat1D

This doc concerns solving 1D heat conduction problems, where the left tip is convected to air, and right tip is adiabatic, and identifies optimal simulation parameters: CFL number and grid resolution. The metric for convergence check is the heat flux into the air (ie, gradient at the left tip) at the final time step.

## Finding Convergent CFL Number

**Notes:**
The CFL (Courant-Friedrichs-Lewy) condition establishes a relationship between temporal and spatial discretization. For heat transfer problems:
```
dt = CFL * α / dx²
```
where:
- α = thermal diffusivity (k*ρ/cₚ)
- k = thermal conductivity
- ρ = density
- cₚ = specific heat capacity

To determine the maximum stable CFL number for a given grid size, progressively reduce the CFL until convergence is achieved. The dummy solution implements a menthod halving CFL each iteration.

**Example Command:**
```bash
python dummy_sols/heat1d.py --task cfl --profile p1 --initial_cfl 1.0 --initial_n_space 100
```

## Finding Convergent Grid Resolution

**Notes:**
Grid resolution determines spatial discretization (dx = L/n_space). Higher resolution provides finer solutions but increases computational cost. Note that dt automatically adjusts according to the CFL condition.

For a fixed CFL number, double the grid resolution until the solution converges.

**Example Command:**
```bash
python dummy_sols/heat1d.py --task n_space --profile p1 --initial_cfl 1.0 --initial_n_space 10
```

**Parameters:**
| Parameter        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--task`         | Task type (`cfl` or `n_space`)                                              |
| `--profile`      | Physics/numerical profile                                              |
| `--initial_cfl`  | Starting CFL number (halved each iteration for CFL task)                    |
| `--initial_n_space` | Starting grid resolution (doubled each iteration for grid task)          |

## Creating New Test Profiles

To generate new test cases with randomized material properties:

```python
import numpy as np
import yaml

def create_new_profile(profile_name):
    # TODO: Read base configuration from existing p1
    
    # Generate randomized material properties
    log_h_min = np.log10(0.1)
    log_h_max = np.log10(100)
    log_h = np.random.uniform(log_h_min, log_h_max)
    
    properties = {
        'h': round(10**log_h, 2),          # Heat transfer coefficient [W/m²-K]
        'L': round(np.random.uniform(0.1, 0.2), 3),  # Rod length [m]
        'k': round(np.random.uniform(0.5, 1), 2),    # Thermal conductivity [W/m-K]
        'rho': round(np.random.uniform(1000, 2000)),  # Density [kg/m³]
        'cp': round(np.random.uniform(800, 1000)),    # Specific heat [J/kg-K]
        'T_inf': round(np.random.uniform(4, 20)),     # Ambient temp [°C]
        'T_init': round(np.random.uniform(21, 30)),   # Initial temp [°C]
        'record_dt': round(np.random.uniform(1, 4)) * 100  # Recording interval [s]
    }
    
    # TODO: change the dump dir to avoid overwrite
    # dump_dir: "sim_res/heat_steady_2d/p1" -> dump_dir: "sim_res/heat_steady_2d/p2" etc
    # TODO: Implement profile saving logic
    # TODO: Save as p2, p3, etc.
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
