#!/usr/bin/env python3
"""
Analyze energy metrics from cached FEM2D simulations.

This script loads all cached simulation results and analyzes:
1. Energy conservation metrics (coefficient of variation) for each profile
2. Energy differences between adjacent tunable parameter values
3. Ranges of energy scales across different profiles

The goal is to determine appropriate precision thresholds for each profile.
"""

import os
import sys
import yaml
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wrappers.fem2d import get_fem2d_data, compute_energy_metrics
from solvers.utils import format_param_for_path


def load_checkout_config():
    """Load the checkout configuration"""
    config_path = "checkouts/fem2d.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_parameter_sequences(config):
    """Extract the parameter sequences for each target parameter and profile"""
    sequences = {}

    # For nx parameter
    if "nx" in config["target_parameters"]:
        nx_config = config["target_parameters"]["nx"]
        nx_sequences = {}
        for profile in ["p1", "p2", "p3"]:
            initial = nx_config["initial_values"][profile]
            factor = nx_config["multiplication_factor"]
            max_iter = nx_config["max_iteration_num"]

            seq = [initial]
            for _ in range(max_iter - 1):
                seq.append(int(seq[-1] * factor))
            nx_sequences[profile] = seq

        sequences["nx"] = nx_sequences

    # For cfl parameter
    if "cfl" in config["target_parameters"]:
        cfl_config = config["target_parameters"]["cfl"]
        cfl_sequences = {}
        for profile in ["p1", "p2", "p3"]:
            initial = cfl_config["initial_values"][profile]
            factor = cfl_config["multiplication_factor"]
            max_iter = cfl_config["max_iteration_num"]

            seq = [initial]
            for _ in range(max_iter - 1):
                seq.append(seq[-1] * factor)
            cfl_sequences[profile] = seq

        sequences["cfl"] = cfl_sequences

    return sequences


def analyze_nx_parameter(profiles, config):
    """Analyze energy metrics for nx parameter variations"""
    print("\n" + "="*80)
    print("ANALYZING NX PARAMETER")
    print("="*80)

    nx_config = config["target_parameters"]["nx"]
    results = {}

    for profile in profiles:
        print(f"\n--- Profile: {profile} ---")
        results[profile] = {
            "nx_values": [],
            "energy_variations": [],
            "avg_energy_diffs": [],
            "energy_scales": {},
            "metrics_list": []
        }

        # Get nx sequence for this profile
        initial_nx = nx_config["initial_values"][profile]
        factor = nx_config["multiplication_factor"]
        max_iter = nx_config["max_iteration_num"]

        nx_sequence = [initial_nx]
        for _ in range(max_iter - 1):
            nx_sequence.append(int(nx_sequence[-1] * factor))

        # Get non-target parameters (use first value)
        cfl = nx_config["non_target_parameters"]["cfl"][profile][0]
        newton_v_res_tol = nx_config["non_target_parameters"]["newton_v_res_tol"][profile][0]

        print(f"nx sequence: {nx_sequence}")
        print(f"Fixed parameters: cfl={cfl}, newton_v_res_tol={newton_v_res_tol}")

        # Load data for all nx values
        energies_data = []
        for nx in nx_sequence:
            try:
                energies, cost, _ = get_fem2d_data(profile, nx, cfl)

                # Compute metrics with a reference var_threshold
                var_threshold = 0.01  # Use a reference value
                metrics = compute_energy_metrics(energies, var_threshold)

                energies_data.append((nx, energies, cost, metrics))

                results[profile]["nx_values"].append(nx)
                results[profile]["energy_variations"].append(metrics["energy_variation"])
                results[profile]["metrics_list"].append(metrics)

                # Print individual metrics
                print(f"  nx={nx}:")
                print(f"    Energy variation (CoV): {metrics['energy_variation']:.6e}")
                print(f"    Energy conserved (at var_threshold={var_threshold}): {metrics['energy_conserved']}")
                print(f"    Total energy range: [{metrics['tot_min']:.6e}, {metrics['tot_max']:.6e}]")
                print(f"    Kinetic energy range: [{metrics['kin_min']:.6e}, {metrics['kin_max']:.6e}]")
                print(f"    Potential energy range: [{metrics['pot_min']:.6e}, {metrics['pot_max']:.6e}]")
                print(f"    Gravitational energy range: [{metrics['gra_min']:.6e}, {metrics['gra_max']:.6e}]")
                print(f"    Cost: {cost}")

            except Exception as e:
                print(f"  ERROR loading nx={nx}: {e}")
                continue

        # Compute energy differences between adjacent nx values
        print(f"\n  Energy differences between adjacent nx values:")
        for i in range(len(energies_data) - 1):
            nx1, energies1, _, _ = energies_data[i]
            nx2, energies2, _, _ = energies_data[i+1]

            # Calculate relative difference for each energy type
            energy_types = ["kin", "pot", "gra", "tot"]
            rel_diffs = {}

            for energy_type in energy_types:
                energy1 = energies1[energy_type]
                energy2 = energies2[energy_type]

                eps = 1e-12
                rel_diff = np.linalg.norm(energy1 - energy2) / (np.linalg.norm(energy1) + np.linalg.norm(energy2) + eps)
                rel_diffs[energy_type] = rel_diff

            avg_diff = np.mean([rel_diffs[et] for et in ["kin", "pot", "gra"]])
            results[profile]["avg_energy_diffs"].append(avg_diff)

            print(f"    nx={nx1} -> nx={nx2}:")
            print(f"      Relative diffs: kin={rel_diffs['kin']:.6e}, pot={rel_diffs['pot']:.6e}, gra={rel_diffs['gra']:.6e}, tot={rel_diffs['tot']:.6e}")
            print(f"      Average (kin+pot+gra): {avg_diff:.6e}")

        # Store energy scale information
        if energies_data:
            all_kin = np.concatenate([ed[1]["kin"] for ed in energies_data])
            all_pot = np.concatenate([ed[1]["pot"] for ed in energies_data])
            all_gra = np.concatenate([ed[1]["gra"] for ed in energies_data])
            all_tot = np.concatenate([ed[1]["tot"] for ed in energies_data])

            results[profile]["energy_scales"] = {
                "kin_range": [np.min(all_kin), np.max(all_kin)],
                "pot_range": [np.min(all_pot), np.max(all_pot)],
                "gra_range": [np.min(all_gra), np.max(all_gra)],
                "tot_range": [np.min(all_tot), np.max(all_tot)],
            }

    return results


def analyze_cfl_parameter(profiles, config):
    """Analyze energy metrics for cfl parameter variations"""
    print("\n" + "="*80)
    print("ANALYZING CFL PARAMETER")
    print("="*80)

    cfl_config = config["target_parameters"]["cfl"]
    results = {}

    for profile in profiles:
        print(f"\n--- Profile: {profile} ---")
        results[profile] = {
            "cfl_values": [],
            "energy_variations": [],
            "avg_energy_diffs": [],
            "energy_scales": {},
            "metrics_list": []
        }

        # Get cfl sequence for this profile
        initial_cfl = cfl_config["initial_values"][profile]
        factor = cfl_config["multiplication_factor"]
        max_iter = cfl_config["max_iteration_num"]

        cfl_sequence = [initial_cfl]
        for _ in range(max_iter - 1):
            cfl_sequence.append(cfl_sequence[-1] * factor)

        # Get non-target parameters - use first nx value for each profile
        nx_values = cfl_config["non_target_parameters"]["nx"][profile]
        newton_v_res_tol = cfl_config["non_target_parameters"]["newton_v_res_tol"][profile][0]

        print(f"cfl sequence: {cfl_sequence}")
        print(f"Testing with nx values: {nx_values}")
        print(f"Fixed parameter: newton_v_res_tol={newton_v_res_tol}")

        # Analyze for each nx value
        for nx in nx_values:
            print(f"\n  --- nx={nx} ---")

            # Load data for all cfl values
            energies_data = []
            for cfl in cfl_sequence:
                try:
                    energies, cost, _ = get_fem2d_data(profile, nx, cfl)

                    # Compute metrics with a reference var_threshold
                    var_threshold = 0.01  # Use a reference value
                    metrics = compute_energy_metrics(energies, var_threshold)

                    energies_data.append((cfl, energies, cost, metrics))

                    if nx == nx_values[0]:  # Only store for first nx
                        results[profile]["cfl_values"].append(cfl)
                        results[profile]["energy_variations"].append(metrics["energy_variation"])
                        results[profile]["metrics_list"].append(metrics)

                    # Print individual metrics
                    print(f"    cfl={cfl}:")
                    print(f"      Energy variation (CoV): {metrics['energy_variation']:.6e}")
                    print(f"      Energy conserved (at var_threshold={var_threshold}): {metrics['energy_conserved']}")
                    print(f"      Total energy range: [{metrics['tot_min']:.6e}, {metrics['tot_max']:.6e}]")
                    print(f"      Cost: {cost}")

                except Exception as e:
                    print(f"    ERROR loading cfl={cfl}: {e}")
                    continue

            # Compute energy differences between adjacent cfl values
            print(f"\n    Energy differences between adjacent cfl values:")
            for i in range(len(energies_data) - 1):
                cfl1, energies1, _, _ = energies_data[i]
                cfl2, energies2, _, _ = energies_data[i+1]

                # Calculate relative difference for each energy type
                energy_types = ["kin", "pot", "gra", "tot"]
                rel_diffs = {}

                for energy_type in energy_types:
                    energy1 = energies1[energy_type]
                    energy2 = energies2[energy_type]

                    eps = 1e-12
                    rel_diff = np.linalg.norm(energy1 - energy2) / (np.linalg.norm(energy1) + np.linalg.norm(energy2) + eps)
                    rel_diffs[energy_type] = rel_diff

                avg_diff = np.mean([rel_diffs[et] for et in ["kin", "pot", "gra"]])
                if nx == nx_values[0]:  # Only store for first nx
                    results[profile]["avg_energy_diffs"].append(avg_diff)

                print(f"      cfl={cfl1} -> cfl={cfl2}:")
                print(f"        Relative diffs: kin={rel_diffs['kin']:.6e}, pot={rel_diffs['pot']:.6e}, gra={rel_diffs['gra']:.6e}")
                print(f"        Average (kin+pot+gra): {avg_diff:.6e}")

    return results


def generate_recommendations(nx_results, cfl_results, config):
    """Generate precision threshold recommendations based on analysis"""
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR PRECISION THRESHOLDS")
    print("="*80)

    profiles = ["p1", "p2", "p3"]

    print("\n--- Current Precision Levels ---")
    for level, thresholds in config["precision_levels"].items():
        print(f"{level:8s}: energy_tolerance={thresholds['energy_tolerance']:.4f}, var_threshold={thresholds['var_threshold']:.6f}")

    print("\n--- Analysis Summary ---")
    for profile in profiles:
        print(f"\nProfile {profile}:")

        # NX results
        if profile in nx_results and nx_results[profile]["energy_variations"]:
            nx_variations = nx_results[profile]["energy_variations"]
            nx_diffs = nx_results[profile]["avg_energy_diffs"]

            print(f"  NX parameter:")
            print(f"    Energy variation (CoV) range: [{min(nx_variations):.6e}, {max(nx_variations):.6e}]")
            if nx_diffs:
                print(f"    Adjacent energy diff range: [{min(nx_diffs):.6e}, {max(nx_diffs):.6e}]")

        # CFL results
        if profile in cfl_results and cfl_results[profile]["energy_variations"]:
            cfl_variations = cfl_results[profile]["energy_variations"]
            cfl_diffs = cfl_results[profile]["avg_energy_diffs"]

            print(f"  CFL parameter:")
            print(f"    Energy variation (CoV) range: [{min(cfl_variations):.6e}, {max(cfl_variations):.6e}]")
            if cfl_diffs:
                print(f"    Adjacent energy diff range: [{min(cfl_diffs):.6e}, {max(cfl_diffs):.6e}]")

    print("\n--- Recommended Precision Thresholds ---")
    print("\nBased on the observed energy variations and differences,")
    print("here are suggested profile-specific precision thresholds:\n")

    recommendations = {}

    for profile in profiles:
        print(f"{profile}:")

        # Collect all variations and diffs for this profile
        all_variations = []
        all_diffs = []

        if profile in nx_results:
            all_variations.extend(nx_results[profile]["energy_variations"])
            all_diffs.extend(nx_results[profile]["avg_energy_diffs"])

        if profile in cfl_results:
            all_variations.extend(cfl_results[profile]["energy_variations"])
            all_diffs.extend(cfl_results[profile]["avg_energy_diffs"])

        if all_variations and all_diffs:
            max_variation = max(all_variations)
            min_diff = min(all_diffs)

            # Recommend var_threshold to be slightly above max observed variation
            # Recommend energy_tolerance to be slightly below min observed diff

            var_threshold_high = max_variation * 2.0
            var_threshold_medium = max_variation * 5.0
            var_threshold_low = max_variation * 10.0

            energy_tolerance_high = min_diff * 0.5
            energy_tolerance_medium = min_diff * 1.0
            energy_tolerance_low = min_diff * 2.0

            recommendations[profile] = {
                "high": {
                    "var_threshold": var_threshold_high,
                    "energy_tolerance": energy_tolerance_high
                },
                "medium": {
                    "var_threshold": var_threshold_medium,
                    "energy_tolerance": energy_tolerance_medium
                },
                "low": {
                    "var_threshold": var_threshold_low,
                    "energy_tolerance": energy_tolerance_low
                }
            }

            print(f"  High precision:")
            print(f"    var_threshold: {var_threshold_high:.6e} (2x max variation: {max_variation:.6e})")
            print(f"    energy_tolerance: {energy_tolerance_high:.6e} (0.5x min diff: {min_diff:.6e})")
            print(f"  Medium precision:")
            print(f"    var_threshold: {var_threshold_medium:.6e} (5x max variation)")
            print(f"    energy_tolerance: {energy_tolerance_medium:.6e} (1.0x min diff)")
            print(f"  Low precision:")
            print(f"    var_threshold: {var_threshold_low:.6e} (10x max variation)")
            print(f"    energy_tolerance: {energy_tolerance_low:.6e} (2.0x min diff)")

    return recommendations


def main():
    """Main analysis function"""
    print("="*80)
    print("FEM2D ENERGY METRICS ANALYSIS")
    print("="*80)

    # Load configuration
    config = load_checkout_config()
    profiles = config["profiles"]["active_profiles"]

    print(f"\nActive profiles: {profiles}")
    print(f"Target parameters: {list(config['target_parameters'].keys())}")

    # Analyze nx parameter
    nx_results = {}
    if "nx" in config["target_parameters"]:
        nx_results = analyze_nx_parameter(profiles, config)

    # Analyze cfl parameter
    cfl_results = {}
    if "cfl" in config["target_parameters"]:
        cfl_results = analyze_cfl_parameter(profiles, config)

    # Generate recommendations
    recommendations = generate_recommendations(nx_results, cfl_results, config)

    # Save recommendations to file
    output_path = "outputs/energy_metrics_analysis.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        "nx_results": {
            profile: {
                "nx_values": results["nx_values"],
                "energy_variations": results["energy_variations"],
                "avg_energy_diffs": results["avg_energy_diffs"],
                "energy_scales": results["energy_scales"]
            }
            for profile, results in nx_results.items()
        },
        "cfl_results": {
            profile: {
                "cfl_values": results["cfl_values"],
                "energy_variations": results["energy_variations"],
                "avg_energy_diffs": results["avg_energy_diffs"]
            }
            for profile, results in cfl_results.items()
        },
        "recommendations": recommendations
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Analysis complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
