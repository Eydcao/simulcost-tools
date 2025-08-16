import argparse
import numpy as np
import yaml

# Append abs path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrappers import *

# Grid search version for mesh_x
def grid_search_mesh_x(profile, boundary_condition, mesh_x_values, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold, length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance):
    param_history = []
    cost_history = []
    converged = False
    best_mesh_x = None

    for mesh_x in mesh_x_values:
        print(f"\nRunning simulation with mesh_x = {mesh_x}")
        cost_i, _ = run_sim_ns_channel_2d(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold)
        cost_history.append(cost_i)
        param_history.append({
            "mesh_x": mesh_x,
            "mesh_y": mesh_y,
            "omega_u": omega_u,
            "omega_v": omega_v,
            "omega_p": omega_p,
            "diff_u_threshold": diff_u_threshold,
            "diff_v_threshold": diff_v_threshold,
            "res_iter_v_threshold": res_iter_v_threshold
        })

        if len(param_history) > 1:
            prev_mesh_x = param_history[-2]["mesh_x"]
            is_converged, _, _, _, _, _ = compare_res_ns_channel_2d(
                profile, boundary_condition, prev_mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
            )
            if is_converged:
                print(f"Convergence achieved between mesh_x {prev_mesh_x} and {mesh_x}")
                best_mesh_x = param_history[-1]["mesh_x"] # The finer of the two meshes that converged
                converged = True
                break
            else:
                print(f"No convergence between mesh_x {prev_mesh_x} and {mesh_x}")

    if not converged and param_history:
        best_mesh_x = param_history[-1]["mesh_x"]
    return bool(converged), best_mesh_x, cost_history, param_history

# Grid search version for mesh_y
def grid_search_mesh_y(profile, boundary_condition, mesh_x, mesh_y_values, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold, length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance):
    param_history = []
    cost_history = []
    converged = False
    best_mesh_y = None

    for mesh_y in mesh_y_values:
        print(f"\nRunning simulation with mesh_y = {mesh_y}")
        cost_i, _ = run_sim_ns_channel_2d(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold)
        cost_history.append(cost_i)
        param_history.append({
            "mesh_x": mesh_x,
            "mesh_y": mesh_y,
            "omega_u": omega_u,
            "omega_v": omega_v,
            "omega_p": omega_p,
            "diff_u_threshold": diff_u_threshold,
            "diff_v_threshold": diff_v_threshold,
            "res_iter_v_threshold": res_iter_v_threshold
        })
        if len(param_history) > 1:
            prev_mesh_y = param_history[-2]["mesh_y"]
            is_converged, _, _, _, _, _ = compare_res_ns_channel_2d(
                profile, boundary_condition, mesh_x, prev_mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
            )
            if is_converged:
                print(f"Convergence achieved between mesh_y {prev_mesh_y} and {mesh_y}")
                best_mesh_y = param_history[-1]["mesh_y"] # The finer of the two meshes that converged
                converged = True
                break
            else:
                print(f"No convergence between mesh_y {prev_mesh_y} and {mesh_y}")
    if not converged and param_history:
        best_mesh_y = param_history[-1]["mesh_y"]
    return bool(converged), best_mesh_y, cost_history, param_history


def grid_search_omega_u(profile, boundary_condition, mesh_x, mesh_y, omega_u_values, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold, length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance):
    """
    Grid search for omega_u: run over all omega_u_values, compare results for convergence.
    Returns: (converged, best_omega_u, cost_history, param_history)
    """
    param_history = []
    cost_history = []
    converged = False
    best_omega_u = None

    for omega_u in omega_u_values:
        print(f"\nRunning simulation with omega_u = {omega_u}")
        cost_i, num_steps = run_sim_ns_channel_2d(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold)
        cost_history.append(cost_i)
        param_history.append({
            "mesh_x": mesh_x,
            "mesh_y": mesh_y,
            "omega_u": omega_u,
            "omega_v": omega_v,
            "omega_p": omega_p,
            "diff_u_threshold": diff_u_threshold,
            "diff_v_threshold": diff_v_threshold,
            "res_iter_v_threshold": res_iter_v_threshold
        })
        print(f"Cost for omega_u={omega_u}: {cost_i}")
        # After at least two, compare
        if len(param_history) > 1:
            prev_omega_u = param_history[-2]["omega_u"]
            is_converged, _, _, _, _, _ = compare_res_ns_channel_2d(
                profile, boundary_condition, mesh_x, mesh_y, prev_omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
            )
            if is_converged:
                print(f"Convergence achieved between omega_u {prev_omega_u} and {omega_u}")
                best_omega_u = param_history[-1]["omega_u"]
                converged = True
                break
            else:
                print(f"No convergence between omega_u {prev_omega_u} and {omega_u}")

    if converged:
        print(f"\nConvergent omega_u found: {best_omega_u}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(param_history) > 0:
            best_omega_u = param_history[-1]["omega_u"]
            print(f"Last tested omega_u: {best_omega_u}")
        else:
            best_omega_u = None
    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")
    return bool(converged), best_omega_u, cost_history, param_history


def grid_search_omega_v(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v_values, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold, length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance):
    """
    Grid search for omega_v: run over all omega_v_values, compare results for convergence.
    Returns: (converged, best_omega_v, cost_history, param_history)
    """
    param_history = []
    cost_history = []
    converged = False
    best_omega_v = None

    for omega_v in omega_v_values:
        print(f"\nRunning simulation with omega_v = {omega_v}")
        cost_i, num_steps = run_sim_ns_channel_2d(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold)
        cost_history.append(cost_i)
        param_history.append({
            "mesh_x": mesh_x,
            "mesh_y": mesh_y,
            "omega_u": omega_u,
            "omega_v": omega_v,
            "omega_p": omega_p,
            "diff_u_threshold": diff_u_threshold,
            "diff_v_threshold": diff_v_threshold,
            "res_iter_v_threshold": res_iter_v_threshold
        })
        print(f"Cost for omega_v={omega_v}: {cost_i}")
        if len(param_history) > 1:
            prev_omega_v = param_history[-2]["omega_v"]
            is_converged, _, _, _, _, _ = compare_res_ns_channel_2d(
                profile, boundary_condition, mesh_x, mesh_y, omega_u, prev_omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
            )
            if is_converged:
                print(f"Convergence achieved between omega_v {prev_omega_v} and {omega_v}")
                best_omega_v = param_history[-1]["omega_v"]
                converged = True
                break
            else:
                print(f"No convergence between omega_v {prev_omega_v} and {omega_v}")

    if converged:
        print(f"\nConvergent omega_v found: {best_omega_v}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(param_history) > 0:
            best_omega_v = param_history[-1]["omega_v"]
            print(f"Last tested omega_v: {best_omega_v}")
        else:
            best_omega_v = None
    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")
    return bool(converged), best_omega_v, cost_history, param_history


def grid_search_omega_p(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p_values, diff_u_threshold, diff_v_threshold, res_iter_v_threshold, length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance):
    """
    Grid search for omega_p: run over all omega_p_values, compare results for convergence.
    Returns: (converged, best_omega_p, cost_history, param_history)
    """
    param_history = []
    cost_history = []
    converged = False
    best_omega_p = None

    for omega_p in omega_p_values:
        print(f"\nRunning simulation with omega_p = {omega_p}")
        cost_i, num_steps = run_sim_ns_channel_2d(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold)
        cost_history.append(cost_i)
        param_history.append({
            "mesh_x": mesh_x,
            "mesh_y": mesh_y,
            "omega_u": omega_u,
            "omega_v": omega_v,
            "omega_p": omega_p,
            "diff_u_threshold": diff_u_threshold,
            "diff_v_threshold": diff_v_threshold,
            "res_iter_v_threshold": res_iter_v_threshold
        })
        print(f"Cost for omega_p={omega_p}: {cost_i}")
        if len(param_history) > 1:
            prev_omega_p = param_history[-2]["omega_p"]
            is_converged, _, _, _, _, _ = compare_res_ns_channel_2d(
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, prev_omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
            )
            if is_converged:
                print(f"Convergence achieved between omega_p {prev_omega_p} and {omega_p}")
                best_omega_p = param_history[-1]["omega_p"]
                converged = True
                break
            else:
                print(f"No convergence between omega_p {prev_omega_p} and {omega_p}")

    if converged:
        print(f"\nConvergent omega_p found: {best_omega_p}")
    else:
        print("\nMaximum iterations reached without convergence")
        if len(param_history) > 0:
            best_omega_p = param_history[-1]["omega_p"]
            print(f"Last tested omega_p: {best_omega_p}")
        else:
            best_omega_p = None
    print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")
    return bool(converged), best_omega_p, cost_history, param_history

# Grid search version for diff_u_threshold
def grid_search_diff_u_threshold(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_values, diff_v_threshold, res_iter_v_threshold, length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance):
    param_history = []
    cost_history = []
    converged = False
    best_diff_u_threshold = None
    for diff_u_threshold in diff_u_values:
        print(f"\nRunning simulation with diff_u_threshold = {diff_u_threshold}")
        cost_i, _ = run_sim_ns_channel_2d(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold)
        cost_history.append(cost_i)
        param_history.append({
            "mesh_x": mesh_x,
            "mesh_y": mesh_y,
            "omega_u": omega_u,
            "omega_v": omega_v,
            "omega_p": omega_p,
            "diff_u_threshold": diff_u_threshold,
            "diff_v_threshold": diff_v_threshold,
            "res_iter_v_threshold": res_iter_v_threshold
        })
        if len(param_history) > 1:
            prev_diff_u_threshold = param_history[-2]["diff_u_threshold"]
            is_converged, _, _, _, _, _ = compare_res_ns_channel_2d(
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, prev_diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
            )
            if is_converged:
                print(f"Convergence achieved between diff_u_threshold {prev_diff_u_threshold} and {diff_u_threshold}")
                best_diff_u_threshold = param_history[-1]["diff_u_threshold"]
                converged = True
                break
            else:
                print(f"No convergence between diff_u_threshold {prev_diff_u_threshold} and {diff_u_threshold}")
    if not converged and param_history:
        best_diff_u_threshold = param_history[-1]["diff_u_threshold"]
    return bool(converged), best_diff_u_threshold, cost_history, param_history

# Grid search version for diff_v_threshold
def grid_search_diff_v_threshold(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_values, res_iter_v_threshold, length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance):
    param_history = []
    cost_history = []
    converged = False
    best_diff_v_threshold = None
    for diff_v_threshold in diff_v_values:
        print(f"\nRunning simulation with diff_v_threshold = {diff_v_threshold}")
        cost_i, _ = run_sim_ns_channel_2d(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold)
        cost_history.append(cost_i)
        param_history.append({
            "mesh_x": mesh_x,
            "mesh_y": mesh_y,
            "omega_u": omega_u,
            "omega_v": omega_v,
            "omega_p": omega_p,
            "diff_u_threshold": diff_u_threshold,
            "diff_v_threshold": diff_v_threshold,
            "res_iter_v_threshold": res_iter_v_threshold
        })
        if len(param_history) > 1:
            prev_diff_v_threshold = param_history[-2]["diff_v_threshold"]
            is_converged, _, _, _, _, _ = compare_res_ns_channel_2d(
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, prev_diff_v_threshold, res_iter_v_threshold,
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
            )
            if is_converged:
                print(f"Convergence achieved between diff_v_threshold {prev_diff_v_threshold} and {diff_v_threshold}")
                best_diff_v_threshold = param_history[-1]["diff_v_threshold"]
                converged = True
                break
            else:
                print(f"No convergence between diff_v_threshold {prev_diff_v_threshold} and {diff_v_threshold}")
    if not converged and param_history:
        best_diff_v_threshold = param_history[-1]["diff_v_threshold"]
    return bool(converged), best_diff_v_threshold, cost_history, param_history


# def find_optimal_res_iter_v_threshold(profile, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, initial_res_iter_v_threshold, length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance, max_iter):
#     """Decrease res_iter_v_threshold by factors of 10 until convergence."""
#     param_history = []
#     cost_history = []
#     current_res_iter_v_threshold = initial_res_iter_v_threshold
#     best_res_iter_v_threshold = None
#     converged = False
#     for i in range(max_iter):
#         print(f"\nRunning simulation with res_iter_v_threshold = {current_res_iter_v_threshold}")
#         cost_i, num_steps = run_sim_ns_channel_2d(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, current_res_iter_v_threshold)
#         cost_history.append(cost_i)
#         param_history.append({
#             "mesh_x": mesh_x,
#             "mesh_y": mesh_y,
#             "omega_u": omega_u,
#             "omega_v": omega_v,
#             "omega_p": omega_p,
#             "diff_u_threshold": diff_u_threshold,
#             "diff_v_threshold": diff_v_threshold,
#             "res_iter_v_threshold": current_res_iter_v_threshold
#         })
#         if len(param_history) > 1:
#             prev_res_iter_v_threshold = param_history[-2]["res_iter_v_threshold"]
#             is_converged, _, _, _, _, _ = compare_res_ns_channel_2d(
#                 profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, prev_res_iter_v_threshold,
#                 profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, current_res_iter_v_threshold,
#                 length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
#             )
#             if is_converged:
#                 print(f"Convergence achieved between res_iter_v_threshold {prev_res_iter_v_threshold} and {current_res_iter_v_threshold}")
#                 best_res_iter_v_threshold = prev_res_iter_v_threshold
#                 converged = True
#                 break
#             else:
#                 print(f"No convergence between res_iter_v_threshold {prev_res_iter_v_threshold} and {current_res_iter_v_threshold}")
#         next_res_iter_v_threshold = current_res_iter_v_threshold / 10
#         current_res_iter_v_threshold = next_res_iter_v_threshold
#     if converged:
#         print(f"\nConvergent res_iter_v_threshold found: {best_res_iter_v_threshold}")
#     else:
#         print("\nMaximum iterations reached without convergence")
#         if len(param_history) > 0:
#             best_res_iter_v_threshold = param_history[-1]["res_iter_v_threshold"]
#             print(f"Smallest tested res_iter_v_threshold: {best_res_iter_v_threshold}")
#         else:
#             best_res_iter_v_threshold = None
#     print(f"Cost history: {cost_history}, total cost: {sum(cost_history)}")
#     return bool(converged), best_res_iter_v_threshold, cost_history, param_history


def grid_search_res_iter_v_threshold(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_values, length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance):
    param_history = []
    cost_history = []
    converged = False
    best_res_iter_v_threshold = None
    for res_iter_v_threshold in res_iter_v_values:
        print(f"\nRunning simulation with res_iter_v_threshold = {res_iter_v_threshold}")
        cost_i, _ = run_sim_ns_channel_2d(profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold)
        cost_history.append(cost_i)
        param_history.append({
            "mesh_x": mesh_x,
            "mesh_y": mesh_y,
            "omega_u": omega_u,
            "omega_v": omega_v,
            "omega_p": omega_p,
            "diff_u_threshold": diff_u_threshold,
            "diff_v_threshold": diff_v_threshold,
            "res_iter_v_threshold": res_iter_v_threshold
        })
        if len(param_history) > 1:
            prev_res_iter_v_threshold = param_history[-2]["res_iter_v_threshold"]
            is_converged, _, _, _, _, _ = compare_res_ns_channel_2d(
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, prev_res_iter_v_threshold,
                profile, boundary_condition, mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold,
                length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
            )
            if is_converged:
                print(f"Convergence achieved between res_iter_v_threshold {prev_res_iter_v_threshold} and {res_iter_v_threshold}")
                best_res_iter_v_threshold = param_history[-1]["res_iter_v_threshold"]
                converged = True
                break
            else:
                print(f"No convergence between res_iter_v_threshold {prev_res_iter_v_threshold} and {res_iter_v_threshold}")
    if not converged and param_history:
        best_res_iter_v_threshold = param_history[-1]["res_iter_v_threshold"]
    return bool(converged), best_res_iter_v_threshold, cost_history, param_history

# Rule-based parameter search functions
PARAM_RULES = {
    'mesh_x': lambda x: x + 25,
    'mesh_y': lambda x: x + 10,
    'omega_u': lambda x: x + 0.1,
    'omega_v': lambda x: x + 0.1,
    'omega_p': lambda x: x + 0.1,
    'diff_u_threshold': lambda x: x / 10,
    'diff_v_threshold': lambda x: x / 10,
    'res_iter_v_threshold': lambda x: x / 10,
}

def rule_based_search(profile, param_name, initial_value, max_iter, length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance, **fixed_params):
    """
    Rule-based parameter search
    
    Args:
        profile: Profile name
        param_name: Parameter name to adjust
        initial_value: Initial parameter value
        max_iter: Maximum iterations
        **fixed_params: Other fixed parameters
    """
    if param_name not in PARAM_RULES:
        raise ValueError(f"No rule defined for parameter: {param_name}")
    
    rule_func = PARAM_RULES[param_name]
    param_history = []
    cost_history = []
    converged = False
    current_value = initial_value
    
    for iteration in range(max_iter):
        print(f"\nIteration {iteration+1}: Running with {param_name} = {current_value}")
        
        # Update parameter dictionary
        params = {
            'mesh_x': fixed_params.get('mesh_x', 100),
            'mesh_y': fixed_params.get('mesh_y', 20),
            'omega_u': fixed_params.get('omega_u', 0.5),
            'omega_v': fixed_params.get('omega_v', 0.5),
            'omega_p': fixed_params.get('omega_p', 0.5),
            'diff_u_threshold': fixed_params.get('diff_u_threshold', 1e-4),
            'diff_v_threshold': fixed_params.get('diff_v_threshold', 1e-4),
            'res_iter_v_threshold': fixed_params.get('res_iter_v_threshold', 1e-4),
        }
        params[param_name] = current_value
        
        cost_i, _ = run_sim_ns_channel_2d(
            profile, boundary_condition, 
            params['mesh_x'], params['mesh_y'], 
            params['omega_u'], params['omega_v'], params['omega_p'],
            params['diff_u_threshold'], params['diff_v_threshold'], params['res_iter_v_threshold']
        )
        cost_history.append(cost_i)
        param_history.append(params.copy())
        
        # Check convergence
        if len(param_history) > 1:
            prev_params = param_history[-2]
            is_converged, _, _, _, _, _ = compare_res_ns_channel_2d(
                profile, boundary_condition, 
                prev_params['mesh_x'], prev_params['mesh_y'], 
                prev_params['omega_u'], prev_params['omega_v'], prev_params['omega_p'],
                prev_params['diff_u_threshold'], prev_params['diff_v_threshold'], prev_params['res_iter_v_threshold'],
                profile, boundary_condition,
                params['mesh_x'], params['mesh_y'], 
                params['omega_u'], params['omega_v'], params['omega_p'],
                params['diff_u_threshold'], params['diff_v_threshold'], params['res_iter_v_threshold'],
                length, breadth, mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance
            )
            if is_converged:
                print(f"Convergence achieved between {param_name} {prev_params[param_name]} and {current_value}")
                converged = True
                break
            else:
                print(f"No convergence between {param_name} {prev_params[param_name]} and {current_value}")
        
        # Apply rule to generate next value
        if iteration < max_iter - 1:  # Don't generate next value on last iteration
            current_value = rule_func(current_value)
    
    return bool(converged), current_value, cost_history, param_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal parameters for ns_channel_2d simulation")

    # Search mode selection
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "mesh_x",
            "mesh_y",
            "omega_u",
            "omega_v",
            "omega_p",
            "diff_u_threshold",
            "diff_v_threshold",
            "res_iter_v_threshold",
        ],
        required=True,
        help="Choose which parameter to search",
    )
    
    parser.add_argument(
        "--search_mode",
        type=str,
        choices=["grid", "rule"],
        default="rule",
        help="Choose search mode: grid (fixed list) or rule (rule-based)",
    )

    # Profile choice
    parser.add_argument("--profile", type=str, default="p1", help="Name of the simulation profile")

    # Parameter values
    parser.add_argument("--mesh_x", type=int, default=100, help="Mesh size in x direction")
    parser.add_argument("--mesh_y", type=int, default=20, help="Mesh size in y direction")
    parser.add_argument("--omega_u", type=float, default=0.5, help="Relaxation factor for u velocity")
    parser.add_argument("--omega_v", type=float, default=0.5, help="Relaxation factor for v velocity")
    parser.add_argument("--omega_p", type=float, default=0.5, help="Relaxation factor for pressure")
    parser.add_argument("--diff_u_threshold", type=float, default=1e-4, help="Threshold for u velocity difference")
    parser.add_argument("--diff_v_threshold", type=float, default=1e-4, help="Threshold for v velocity difference")
    parser.add_argument("--res_iter_v_threshold", type=float, default=1e-4, help="Threshold for residual iteration v")

    # Fixed parameters
    parser.add_argument("--max_iter", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--length", type=float, default=20.0, help="Length of the channel")
    parser.add_argument("--breadth", type=float, default=1.0, help="Width of the channel")
    parser.add_argument("--mass_tolerance", type=float, default=1e-8, help="Tolerance for outer convergence check")
    parser.add_argument("--u_rmse_tolerance", type=float, default=1e-3, help="Tolerance for convergence checks")
    parser.add_argument("--v_rmse_tolerance", type=float, default=1e-3, help="Tolerance for convergence checks")
    parser.add_argument("--p_rmse_tolerance", type=float, default=1e-3, help="Tolerance for convergence checks")

    args = parser.parse_args()

    # Read boundary_condition from profile config file
    global boundary_condition
    config_path = f"run_configs/ns_channel_2d/{args.profile}.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'boundary_condition' in config:
                boundary_condition = config['boundary_condition']
            else:
                raise ValueError(f"boundary_condition not found in config file {config_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_path} not found. Please ensure the profile '{args.profile}' exists.")

    # Choose search method based on search_mode
    if args.search_mode == "rule":
        print(f"\n=== Starting rule-based {args.task} search ===")
        fixed_params = {
            'mesh_x': args.mesh_x,
            'mesh_y': args.mesh_y,
            'omega_u': args.omega_u,
            'omega_v': args.omega_v,
            'omega_p': args.omega_p,
            'diff_u_threshold': args.diff_u_threshold,
            'diff_v_threshold': args.diff_v_threshold,
            'res_iter_v_threshold': args.res_iter_v_threshold,
        }
        initial_value = getattr(args, args.task)
        converged, best_param, cost_history, param_history = rule_based_search(
            args.profile, args.task, initial_value, args.max_iter,
            args.length, args.breadth, args.mass_tolerance, 
            args.u_rmse_tolerance, args.v_rmse_tolerance, args.p_rmse_tolerance,
            **fixed_params
        )
        total_cost = cost_history
        param_name = args.task
    elif args.task == "mesh_x":
        print("\n=== Starting mesh_x search ===")
        mesh_x_values = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
        converged, best_param, cost_history, param_history = grid_search_mesh_x(
            args.profile, boundary_condition, mesh_x_values, args.mesh_y, args.omega_u, args.omega_v, args.omega_p, args.diff_u_threshold, args.diff_v_threshold, args.res_iter_v_threshold,
            args.length, args.breadth, args.mass_tolerance, args.u_rmse_tolerance, args.v_rmse_tolerance, args.p_rmse_tolerance
        )
        total_cost = cost_history
        param_name = "mesh_x"
    elif args.task == "mesh_y":
        print("\n=== Starting mesh_y search ===")
        mesh_y_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        converged, best_param, cost_history, param_history = grid_search_mesh_y(
            args.profile, boundary_condition, args.mesh_x, mesh_y_values, args.omega_u, args.omega_v, args.omega_p, args.diff_u_threshold, args.diff_v_threshold, args.res_iter_v_threshold,
            args.length, args.breadth, args.mass_tolerance, args.u_rmse_tolerance, args.v_rmse_tolerance, args.p_rmse_tolerance
        )
        total_cost = cost_history
        param_name = "mesh_y"
    elif args.task == "omega_u":
        print("\n=== Starting omega_u search ===")
        omega_u_values = [0.1 * i for i in range(1, 11)]
        converged, best_param, cost_history, param_history = grid_search_omega_u(
            args.profile, boundary_condition, args.mesh_x, args.mesh_y, omega_u_values, args.omega_v, args.omega_p, args.diff_u_threshold, args.diff_v_threshold, args.res_iter_v_threshold,
            args.length, args.breadth, args.mass_tolerance, args.u_rmse_tolerance, args.v_rmse_tolerance, args.p_rmse_tolerance
        )
        total_cost = cost_history
        param_name = "omega_u"
    elif args.task == "omega_v":
        print("\n=== Starting omega_v search ===")
        omega_v_values = [0.1 * i for i in range(1, 11)]
        converged, best_param, cost_history, param_history = grid_search_omega_v(
            args.profile, boundary_condition, args.mesh_x, args.mesh_y, args.omega_u, omega_v_values, args.omega_p, args.diff_u_threshold, args.diff_v_threshold, args.res_iter_v_threshold,
            args.length, args.breadth, args.mass_tolerance, args.u_rmse_tolerance, args.v_rmse_tolerance, args.p_rmse_tolerance
        )
        total_cost = cost_history
        param_name = "omega_v"
    elif args.task == "omega_p":
        print("\n=== Starting omega_p search ===")
        omega_p_values = [0.1 * i for i in range(1, 11)]
        converged, best_param, cost_history, param_history = grid_search_omega_p(
            args.profile, boundary_condition, args.mesh_x, args.mesh_y, args.omega_u, args.omega_v, omega_p_values, args.diff_u_threshold, args.diff_v_threshold, args.res_iter_v_threshold,
            args.length, args.breadth, args.mass_tolerance, args.u_rmse_tolerance, args.v_rmse_tolerance, args.p_rmse_tolerance
        )
        total_cost = cost_history
        param_name = "omega_p"
    elif args.task == "diff_u_threshold":
        print("\n=== Starting diff_u_threshold search ===")
        diff_u_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
        converged, best_param, cost_history, param_history = grid_search_diff_u_threshold(
            args.profile, boundary_condition, args.mesh_x, args.mesh_y, args.omega_u, args.omega_v, args.omega_p, diff_u_values, args.diff_v_threshold, args.res_iter_v_threshold,
            args.length, args.breadth, args.mass_tolerance, args.u_rmse_tolerance, args.v_rmse_tolerance, args.p_rmse_tolerance
        )
        total_cost = cost_history
        param_name = "diff_u_threshold"
    elif args.task == "diff_v_threshold":
        print("\n=== Starting diff_v_threshold search ===")
        diff_v_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
        converged, best_param, cost_history, param_history = grid_search_diff_v_threshold(
            args.profile, boundary_condition, args.mesh_x, args.mesh_y, args.omega_u, args.omega_v, args.omega_p, args.diff_u_threshold, diff_v_values, args.res_iter_v_threshold,
            args.length, args.breadth, args.mass_tolerance, args.u_rmse_tolerance, args.v_rmse_tolerance, args.p_rmse_tolerance
        )
        total_cost = cost_history
        param_name = "diff_v_threshold"
    elif args.task == "res_iter_v_threshold":
        print("\n=== Starting res_iter_v_threshold search ===")
        res_iter_v_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
        converged, best_param, cost_history, param_history = grid_search_res_iter_v_threshold(
            args.profile, boundary_condition, args.mesh_x, args.mesh_y, args.omega_u, args.omega_v, args.omega_p, args.diff_u_threshold, args.diff_v_threshold, res_iter_v_values,
            args.length, args.breadth, args.mass_tolerance, args.u_rmse_tolerance, args.v_rmse_tolerance, args.p_rmse_tolerance
        )
        total_cost = cost_history
        param_name = "res_iter_v_threshold"
    else:
        best_param = None
        total_cost = None
        param_name = args.task
    if best_param is not None:
        print(f"\nRecommended {param_name}: {best_param}, total cost: {total_cost}")
    else:
        print(f"\nNo convergent {param_name} found within the given iterations, total cost: {total_cost}")