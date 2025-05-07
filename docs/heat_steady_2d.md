# Steady State Heat Transfer in 2D

This document describes solving 2D steady-state heat transfer problems using the Jacobi iteration method with Successive Over-Relaxation (SOR). The simulation models a square plate with fixed boundary conditions: the top boundary is held at temperature 1.0, while all other boundaries are held at temperature 0.0.

## Problem Description

The steady-state heat equation in 2D is given by:

∇²T = 0

With boundary conditions:
- T = 1.0 at the top boundary (y = Ly)
- T = 0.0 at all other boundaries (y = 0, x = 0, x = Lx)

The equation is iteratively solved using the Jacobi method with point SOR.

## Parameter Tuning Tasks

### Finding Optimal Grid Resolution (dx) (Task type: 0-shot and iterative)

The grid resolution determines the spatial discretization accuracy. A finer grid (smaller dx) provides more accurate solutions but increases computational cost. In dummy method, we start with an initial dx value and halve it until convergence is achieved between consecutive refinements. The convergence metric is the temperature distribution at the middle (vertical) line

```bash
python dummy_sols/heat_steady_2d.py --task dx --profile p1 --initial_dx 0.01
```

### Finding Optimal Relaxation Factor (relax) (Task type: 0-shot)

The relaxation factor affects convergence speed of the SOR method. Optimal values typically lie between 0 and 2.0. The dummy method perform a grid search within this range and select the value that minimizes computational cost.

Note for relax ratio (SOR) and T_init, a bad choice may lead to NAN/INFITY solution or unable to converge after max number of iterations. Hence, we recorded the "converge" key in the `meta.json`, and we can also check if NAN/INFITY exist in the obtained results from the saved file.

```bash
python dummy_sols/heat_steady_2d.py --task relax --profile p1
```

### Finding Optimal Initial Temperature (T_init) (Task type: 0-shot)

The initial temperature field can affect convergence speed. The dummy method perform a grid search over different initial temperature values and select the one that minimizes computational cost.

Note for relax ratio (SOR) and T_init, a bad choice may lead to NAN/INFITY solution or unable to converge after max number of iterations. Hence, we recorded the "converge" key in the `meta.json`, and we can also check if NAN/INFITY exist in the obtained results from the saved file.

```bash
python dummy_sols/heat_steady_2d.py --task t_init --profile p1
```

### Finding Optimal Error Threshold (Task type: 0-shot and iterative)

The error threshold determines when to stop the Jacobi iteration process.  The convergence metric is the temperature distribution at the middle (vertical) line The dummy method start with a loose threshold and decrease it by factors of 10 until the solution no longer changes significantly.

```bash
python dummy_sols/heat_steady_2d.py --task error_threshold --profile p1 --error_threshold 1e-5
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| dx | Grid spacing (determines resolution) |
| relax | Relaxation factor for SOR method (0.05 <= relax <= 1.95) |
| error_threshold | Convergence criterion (RMSE between iterations) |
| T_init | Initial temperature field value |

## Creating New Test Profiles

To generate new test cases with randomized boundary conditions properties: each of the four walls have a uniform randomized temperature between 0 and 1.

See code for ref:
```bash
python gen_cfgs/heat_steady_2d.py
```