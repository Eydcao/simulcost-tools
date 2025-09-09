import argparse
import numpy as np
import yaml

# Append abs path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrappers import *

# Grid search version for resolution
def grid_search_resolution(profile, boundary_condition, resolution_values, reynolds_num, cfl, relaxation_factor, residual_threshold, total_runtime, norm_rmse_tolerance, other_params=None):
    param_history = []
    cost_history = []
    converged = False
    best_resolution = None

    for i in range(len(resolution_values)):
        resolution = resolution_values[i]
        print(f"\nRunning simulation with resolution = {resolution}")
        cost_i, _ = run_sim_ns_transient_2d(profile, boundary_condition, resolution, reynolds_num, cfl, relaxation_factor, residual_threshold, total_runtime, other_params)
        cost_history.append(cost_i)
        param_history.append({
            "resolution": resolution,
            "boundary_condition": boundary_condition,
            "reynolds_num": reynolds_num,
            "cfl": cfl,
            "relaxation_factor": relaxation_factor,
            "residual_threshold": residual_threshold,
            "total_runtime": total_runtime
        })

        if len(param_history) > 1:
            prev_resolution = param_history[-2]["resolution"]
            is_converged, _ = compare_res_ns_transient_2d(
                profile, boundary_condition, prev_resolution, reynolds_num, cfl, relaxation_factor, residual_threshold, total_runtime,
                profile, boundary_condition, resolution, reynolds_num, cfl, relaxation_factor, residual_threshold, total_runtime,
                norm_rmse_tolerance,
                other_params, other_params
            )
            if is_converged:
                print(f"Convergence achieved between resolution {prev_resolution} and {resolution}")
                best_resolution = param_history[-1]["resolution"]  # The finer of the two resolutions that converged
                converged = True
                break
            else:
                print(f"No convergence between resolution {prev_resolution} and {resolution}")

    if not converged and param_history:
        best_resolution = param_history[-1]["resolution"]
    return bool(converged), best_resolution, cost_history, param_history

# Grid search version for cfl
def grid_search_cfl(profile, boundary_condition, resolution, reynolds_num, cfl_values, relaxation_factor, residual_threshold, total_runtime, norm_rmse_tolerance, other_params=None):
    param_history = []
    cost_history = []
    converged = False
    best_cfl = None

    for cfl in cfl_values:
        print(f"\nRunning simulation with cfl = {cfl}")
        cost_i, _ = run_sim_ns_transient_2d(profile, boundary_condition, resolution, reynolds_num, cfl, relaxation_factor, residual_threshold, total_runtime, other_params)
        cost_history.append(cost_i)
        param_history.append({
            "resolution": resolution,
            "boundary_condition": boundary_condition,
            "reynolds_num": reynolds_num,
            "cfl": cfl,
            "relaxation_factor": relaxation_factor,
            "residual_threshold": residual_threshold,
            "total_runtime": total_runtime
        })

        if len(param_history) > 1:
            prev_cfl = param_history[-2]["cfl"]
            is_converged, _ = compare_res_ns_transient_2d(
                profile, boundary_condition, resolution, reynolds_num, prev_cfl, relaxation_factor, residual_threshold, total_runtime,
                profile, boundary_condition, resolution, reynolds_num, cfl, relaxation_factor, residual_threshold, total_runtime,
                norm_rmse_tolerance,
                other_params, other_params
            )
            if is_converged:
                print(f"Convergence achieved between cfl {prev_cfl} and {cfl}")
                best_cfl = param_history[-1]["cfl"]
                converged = True
                break
            else:
                print(f"No convergence between cfl {prev_cfl} and {cfl}")

    if not converged and param_history:
        best_cfl = param_history[-1]["cfl"]
    return bool(converged), best_cfl, cost_history, param_history

# Grid search version for relaxation_factor
def grid_search_relaxation_factor(profile, boundary_condition, resolution, reynolds_num, cfl, relaxation_factor_values, residual_threshold, total_runtime, norm_rmse_tolerance, other_params=None):
    param_history = []
    cost_history = []
    converged = False
    best_relaxation_factor = None

    for relaxation_factor in relaxation_factor_values:
        print(f"\nRunning simulation with relaxation_factor = {relaxation_factor}")
        cost_i, _ = run_sim_ns_transient_2d(profile, boundary_condition, resolution, reynolds_num, cfl, relaxation_factor, residual_threshold, total_runtime, other_params)
        cost_history.append(cost_i)
        param_history.append({
            "resolution": resolution,
            "boundary_condition": boundary_condition,
            "reynolds_num": reynolds_num,
            "cfl": cfl,
            "relaxation_factor": relaxation_factor,
            "residual_threshold": residual_threshold,
            "total_runtime": total_runtime
        })

        if len(param_history) > 1:
            prev_relaxation_factor = param_history[-2]["relaxation_factor"]
            is_converged, _ = compare_res_ns_transient_2d(
                profile, boundary_condition, resolution, reynolds_num, cfl, prev_relaxation_factor, residual_threshold, total_runtime,
                profile, boundary_condition, resolution, reynolds_num, cfl, relaxation_factor, residual_threshold, total_runtime,
                norm_rmse_tolerance,
                other_params, other_params
            )
            if is_converged:
                print(f"Convergence achieved between relaxation_factor {prev_relaxation_factor} and {relaxation_factor}")
                best_relaxation_factor = param_history[-1]["relaxation_factor"]
                converged = True
                break
            else:
                print(f"No convergence between relaxation_factor {prev_relaxation_factor} and {relaxation_factor}")

    if not converged and param_history:
        best_relaxation_factor = param_history[-1]["relaxation_factor"]
    return bool(converged), best_relaxation_factor, cost_history, param_history

# Grid search version for residual_threshold
def grid_search_residual_threshold(profile, boundary_condition, resolution, reynolds_num, cfl, relaxation_factor, residual_threshold_values, total_runtime, norm_rmse_tolerance, other_params=None):
    param_history = []
    cost_history = []
    converged = False
    best_residual_threshold = None

    for residual_threshold in residual_threshold_values:
        print(f"\nRunning simulation with residual_threshold = {residual_threshold}")
        cost_i, _ = run_sim_ns_transient_2d(profile, boundary_condition, resolution, reynolds_num, cfl, relaxation_factor, residual_threshold, total_runtime, other_params)
        cost_history.append(cost_i)
        param_history.append({
            "resolution": resolution,
            "boundary_condition": boundary_condition,
            "reynolds_num": reynolds_num,
            "cfl": cfl,
            "relaxation_factor": relaxation_factor,
            "residual_threshold": residual_threshold,
            "total_runtime": total_runtime
        })

        if len(param_history) > 1:
            prev_residual_threshold = param_history[-2]["residual_threshold"]
            is_converged, _ = compare_res_ns_transient_2d(
                profile, boundary_condition, resolution, reynolds_num, cfl, relaxation_factor, prev_residual_threshold, total_runtime,
                profile, boundary_condition, resolution, reynolds_num, cfl, relaxation_factor, residual_threshold, total_runtime,
                norm_rmse_tolerance,
                other_params, other_params
            )
            if is_converged:
                print(f"Convergence achieved between residual_threshold {prev_residual_threshold} and {residual_threshold}")
                best_residual_threshold = param_history[-1]["residual_threshold"]
                converged = True
                break
            else:
                print(f"No convergence between residual_threshold {prev_residual_threshold} and {residual_threshold}")

    if not converged and param_history:
        best_residual_threshold = param_history[-1]["residual_threshold"]
    return bool(converged), best_residual_threshold, cost_history, param_history

