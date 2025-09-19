# Developer Guide: Adding New Solvers to CostSci-Tools

This guide helps contributors integrate new PDE solvers into the CostSci-Tools library. The library is designed to benchmark parameter optimization capabilities of a user (can be human or LLM) across different physics-based simulation problems.

## Library Architecture Overview

The library follows a modular design with clear separation of concerns:

### **`solvers/` - Core Numerical Methods**
This is where you implement your actual PDE solver (only if you choose to implement in Python). Your solver should:
- Inherit from the `SIMULATOR` base class in `base_solver.py`
- Implement the core numerical discretization and time-stepping scheme
- Handle spatial discretization, boundary conditions, and solution updates
- Provide methods for calculating adaptive time steps (for transient problems)
- Include output/visualization functionality via the `dump()` method
- Provide methods for cost estimation (eg, using FLOPs complexity)

**Example**: `burgers_1d.py` implements a 2nd order Roe method with minmod limiter for the inviscid Burgers equation.

### **`runners/` - Command Line Interface**
Create a consistent python entry point to run the solver:
- Uses Hydra for configuration management
- Loads parameters from YAML config files
- Instantiates and runs your solver
- Provides a clean command-line interface for users
- **Note**: Your solver does not need to be written in Python, as long as you can wrap the running command into this runner interface (e.g., through `subprocess` in Python) and it saves results that can be used for analysis later.

**Example**: `burgers_1d.py` is a minimal script that loads config and runs the solver.

### **`wrappers/` - High-Level APIs**
Implement wrapper functions that provide:
- **Simulation execution**: Functions to run simulations with given parameters
- **Result management**: Load/save simulation results with caching to avoid redundant computations
- **Results comparison**: Compare solutions between different parameter sets (eg, adjacent spatial resolutions for checking spatial convergence)
- **Self-checking metrics**: Self-checking metrics (e.g., TVD constraint, conservation laws, contains `NaN` or not) to determine if one solution is physically reasonable (**Note**: this means it does not have to be compared between 2 simulation results)

**Example**: `burgers_1d.py` contains functions like `run_sim_burgers_1d()`, `get_res_burgers_1d()`, and `compare_res_burgers_1d()`. For Burgers 1D:
- **Results comparison**: Compares solutions between different CFL values using L∞ norm (max error, crucial near shocks) and RMSE (averaged error in smooth regions). Convergence is achieved when both L∞ ≤ 5e-2 and L² ≤ 5e-3.
- **Self-checking metrics**: Validates mass conservation (mean value constant), energy non-increasing (∫u² decreases), TVD property (total variation doesn't increase), and maximum principle (solution stays within initial bounds).

### **`dummy_sols/` - Brute Force Parameter Searching**
Implement basic scanning algorithms to find optimal parameters for your solver. The search strategy depends on whether the task is 0-shot or iterative (see [Understanding Task Types](#understanding-task-types) section below):
- **Grid search (for zero-shot)**: Exhaust parameter spaces, then choose the parameter with successful run and minimal cost
- **Iterative refinement (for iterative)**: Adaptively refine parameters (e.g., dx) until convergence, find the first convergent parameter
- **Cost optimization**: Find parameters that minimize computational cost while maintaining accuracy

**Example**: `burgers_1d.py` implements iterative refinement for CFL and grid search for k/w.

### **`gen_cfgs/` - Configuration Generation**
Create scripts to generate multiple test configurations:
- **Benchmark problems**: Standard test cases in this domain
- **Randomized parameters**: Generate diverse initial conditions, boundary conditions, or other environmental variables, such as body force terms
- **Parameter variations by human**: As solver developer, please make sure the randomized range is reasonable.

**Example**: `burgers_1d.py` generates different initial conditions for typical benchmarks (sin, rarefaction, sod, double_shock, blast waves). Future work can randomize the parameters of these initial conditions.

### **`run_configs/` - Configuration Files**
Store YAML configuration files that define:
- Physical parameters (domain size, boundary conditions, etc.) **These are usually fixed given a simulation setup**
- Numerical parameters (grid resolution, time step controls, etc.) **These usually contain your tunable parameters**
- Solver-static parameters (frequency for dumping results, etc.) **These are usually fixed given a simulation setup**
- Output settings (sub-directory names)

## Understanding Task Types

The library supports two distinct types of parameter tuning, each testing different aspects of problem-solving capability:

### **0-Shot Tasks** - Testing Domain Knowledge
These tasks evaluate a user's ability to make good initial parameter choices based on theoretical understanding of the problem:

- **Nature**: You must make a single parameter choice upfront, then follow a fixed pipeline
- **Process**: Choose parameter → fixed search/evaluation → get success signal and final cost
- **Cost Calculation**: Determined by the fixed pipeline cost after your initial guess
- **Skills Tested**: Domain expertise, theoretical understanding, parameter intuition
- **Dummy solution**: Grid search -> pick the successful + least cost parameter

**Examples**:
- Choose the optimal `k` value (spatial scheme blending), then automated CFL/dx convergence search. The cost is defined as the total search cost to find the converged solution. Dummy solution will grid search all `k` values and pick the one with the lowest cost + successfully converged results
- Same for optimal `w` value (limiter parameter)
- **Note**: The above examples are 0-shot for `k`/`w` as once a value is picked, the convergence search is a fixed routine (e.g., CFL /= 2 each iteration). This is not iterative for CFL, as the user does not have access to flexibly choose which CFL to try in the next round.

### **Iterative Tasks** - Testing Optimization Strategy  
These tasks evaluate a user's ability to efficiently explore and search within parameter space:

- **Nature**: You can adaptively choose next parameter based on previous observations
- **Process**: Try parameter → observe results → decide next parameter → repeat until success
- **Cost Calculation**: Accumulates across all trials until convergence is achieved
- **Skills Tested**: Search strategy, adaptive optimization, learning from feedback

**Examples**:
- CFL convergence search: You can choose any sequence of CFL values, observing results and adapting your strategy. Dummy solution will apply a fixed shrinking rate (e.g., CFL /= 2) in each iteration

**Note**: Any iterative task can be evaluated as a 0-shot task as well, since good initial parameter intuition will significantly help the searching procedure. Therefore, we also assess the quality of the first parameter choice in iterative tasks to measure domain knowledge alongside search strategy.

## Designing Success Metrics

Success metrics should reflect both the numerical accuracy and physical validity of your solutions. Design metrics that match your solver's characteristics:

### **Numerical Accuracy Metrics**
Based on the Burgers 1D example, consider metrics appropriate for your problem:

- **Smooth solution errors**: Use L²/RMSE norm to represent the averaged error as most of the solution is smooth
- **Sharp discontinuity errors**: Use L∞ norm to capture maximum errors, especially near discontinuities, shocks, or steep gradients
- **Convergence tolerance**: Set appropriate thresholds (e.g., L² ≤ 5e-3, L∞ ≤ 5e-2)

### **Physical Validity Metrics (Self-Checking)**
These verify that your solution satisfies fundamental physical/numerical principles without requiring comparison to reference solutions:

**Conservation Laws**:
- **Mass conservation**: Total integral of conserved quantities should remain constant
- **Energy conservation**: Similarly, but for total kinetic energy

**Stability Properties**:
- **Total Variation Diminishing (TVD)**: For shock-capturing schemes, total variation should not increase
- **Maximum principle**: Solution values should remain within the initial maximum bounds

**Example from Burgers 1D**:
```python
def compute_metrics(u):
    # Mass (mean value should be conserved)
    mass = np.mean(u, axis=1)
    mass_conserved = np.isclose(mass[1:], mass[0], rtol=1e-3)
    
    # Energy (should be non-increasing for Burgers)
    energy = np.mean(u**2, axis=1)
    energy_non_increasing = np.diff(energy) <= 1e-3
    
    # Total Variation (should be non-increasing for TVD schemes)
    TV = np.mean(np.abs(np.diff(u, axis=1)), axis=1)
    TV_non_increasing = np.diff(TV) <= 1e-3
    
    # Maximum principle (solution should stay within initial bounds)
    max_principle_satisfied = check_bounds(u)
```

### **Problem-Specific Considerations**

When designing metrics for your solver:

1. **Identify critical solution features**: What aspects of the solution are most important for your application (overall solution, spectrum, or boundary gradients like heat flux, etc.)? Consider both those that need to be checked by comparing two profiles, or can be self-checked.
2. **Set appropriate tolerances**: Balance numerical precision with computational efficiency. Note that too high precision can lead to infinitely long runtime.
3. **Include multiple error norms**: If needed, apply different norms (L², L∞, etc.) to capture different types of errors
4. **Validate physical/numerical solver properties**: Ensure solutions satisfy fundamental physical laws or numerical solver properties. These usually can be self-checked.

## Implementation Checklist

When adding a new solver:

- [ ] Implement solver class inheriting from `SIMULATOR` (if choosing to implement in Python)
- [ ] Create runner wrapper with Hydra configuration (general for any kind of solver)
- [ ] Implement wrapper functions for simulation management (for users like dummy or LLM to tune with)
- [ ] Define both 0-shot and iterative optimization tasks
- [ ] Design appropriate success metrics for your problem (this will require iteration between this step and next step)
- [ ] Create dummy solutions (this will require iteration between this step and last step)
- [ ] Test with 1 known benchmark problem
- [ ] Expand to diverse test configurations (different ICs, BCs, environments, etc.)
- [ ] Document parameter meanings, task type, and valid ranges

Remember: The goal is to enable fair comparison of different users/algorithms across different numerical methods and simulation cases. Your implementation should be robust, well-documented, and include comprehensive validation metrics.