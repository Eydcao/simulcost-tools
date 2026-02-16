"""
PDE solver implementations for the SimulCost benchmark.

This package contains 12 physics-based numerical solvers, each inheriting
from the SIMULATOR base class in base_solver.py. All solvers track computational
cost and can detect wall time limit violations.

Solvers are imported on-demand using explicit module paths to minimize
startup time and avoid loading unnecessary dependencies.

IMPORTANT: Do NOT add top-level imports here. Each solver should be imported
using its full module path, e.g.:
    from solvers.heat_1d import Heat1D
    from solvers.ns_transient_2d import NSTransient2D

Available solvers:
- heat_1d: Heat1D - 1D transient heat conduction
- heat_steady_2d: SteadyHeat2D - 2D steady-state heat transfer
- burgers_1d: BurgersRoe2 - 1D inviscid Burgers equation
- euler_1d: Euler1D - 1D compressible Euler equations
- euler_2d: (imported separately) - 2D compressible Euler equations
- ns_channel_2d: NSChannel2D - 2D steady Navier-Stokes
- ns_transient_2d: NSTransient2D - 2D transient Navier-Stokes
- diff_react_1d: DiffReact1D - 1D diffusion-reaction
- unstruct_mpm: UNSTRUCT_MPM - Unstructured MPM solid mechanics
- hasegawa_mima_linear: HasegawaMimaLinear - Linear plasma turbulence
- hasegawa_mima_nonlinear: HasegawaMimaNonlinear - Nonlinear plasma turbulence
- fem2d: FEM2D - 2D FEM elasticity
"""
