#!/usr/bin/env python3
"""
Analyze Newton residual values across different dx (spatial resolution) and dt (time step).

This script runs FEM2D simulations with varying nx and cfl values, then logs Newton
residual statistics to understand what absolute tolerance values would be appropriate.

Current residual definition: |Δx|_∞ (infinity norm of displacement correction)
Current convergence: |Δx| < newton_v_res_tol * dt (velocity-like relative tolerance)

Goal: Understand how |Δx| scales with dx and dt to inform absolute tolerance design.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wrappers.fem2d import get_fem2d_data


def collect_residual_data():
    """
    Run simulations across different parameter combinations and collect residual data.

    We'll use verbose=True to capture Newton residual output, but need to modify
    the solver to save residuals to a file instead of just printing.
    """

    print("="*80)
    print("NEWTON RESIDUAL ANALYSIS")
    print("="*80)
    print()
    print("Collecting Newton residual data across different dx and dt values...")
    print()

    # Test configurations
    profiles_to_test = {
        "p1": {"nx_values": [20, 40, 80], "cfl_values": [0.5, 1.0, 2.0]},
        "p2": {"nx_values": [50, 100, 200], "cfl_values": [0.5, 1.0, 2.0]},
        "p3": {"nx_values": [4, 8, 16], "cfl_values": [0.5, 1.0, 2.0]},
    }

    results = []

    for profile, config in profiles_to_test.items():
        print(f"\n--- Profile: {profile} ---")

        for nx in config["nx_values"]:
            for cfl in config["cfl_values"]:
                print(f"  Running: nx={nx}, cfl={cfl}")

                # Run simulation (will use cached if available)
                try:
                    energies, cost, _ = get_fem2d_data(
                        profile=profile,
                        dx=nx,
                        cfl=cfl
                    )

                    # Load meta.json to get actual dt used
                    from solvers.utils import format_param_for_path
                    meta_path = f"sim_res/fem2d/{profile}_nx{nx}_cfl{format_param_for_path(cfl)}_nvrestol{format_param_for_path(1.0)}/meta.json"

                    with open(meta_path, "r") as f:
                        meta = json.load(f)

                    # Estimate dx from domain size and nx
                    # This is approximate - actual dx varies per element
                    if profile == "p1":
                        Lx, Ly = 10.0, 2.0
                    elif profile in ["p2", "p4"]:
                        Lx, Ly = 25.0, 1.0
                    elif profile in ["p3", "p5"]:
                        Lx, Ly = 2.0, 10.0
                    else:
                        Lx, Ly = 10.0, 10.0

                    dx_approx = Lx / nx  # Approximate characteristic mesh size

                    # For now, we don't have dt in meta.json
                    # We'll need to compute it or extract from verbose output
                    # Placeholder: estimate from CFL
                    # dt ≈ cfl * dx / wave_speed
                    # wave_speed ≈ sqrt((lambda + 2*mu) / rho)

                    results.append({
                        "profile": profile,
                        "nx": nx,
                        "cfl": cfl,
                        "dx_approx": dx_approx,
                        "cost": cost,
                        # Will fill in residuals from verbose output or separate logging
                    })

                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

    return results


def run_with_residual_logging():
    """
    Run simulations with modified code to log Newton residuals.

    Since we need to modify the solver code to save residuals, we'll create
    a temporary modified version or use subprocess with verbose output capture.
    """

    print("\n" + "="*80)
    print("RUNNING SIMULATIONS WITH RESIDUAL LOGGING")
    print("="*80)
    print()

    import subprocess

    # Test cases: (profile, nx, cfl)
    test_cases = [
        ("p1", 20, 0.5), ("p1", 20, 1.0), ("p1", 20, 2.0),
        ("p1", 40, 0.5), ("p1", 40, 1.0), ("p1", 40, 2.0),
        ("p1", 80, 0.5), ("p1", 80, 1.0),
        ("p2", 50, 0.5), ("p2", 50, 1.0),
        ("p2", 100, 0.5), ("p2", 100, 1.0),
        ("p3", 4, 0.5), ("p3", 4, 1.0),
        ("p3", 8, 0.5), ("p3", 8, 1.0),
    ]

    residual_data = []

    for profile, nx, cfl in test_cases:
        print(f"\nRunning {profile} with nx={nx}, cfl={cfl}...")

        # Run simulation and capture output
        cmd = [
            sys.executable,
            "runners/fem2d.py",
            f"--config-name={profile}",
            f"nx={nx}",
            f"cfl={cfl}",
            "newton_v_res_tol=1.0",
            "verbose=True"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            # Parse Newton iteration output
            output_lines = result.stdout.split('\n')

            step_num = 0
            current_dt = None
            current_dx = None

            for line in output_lines:
                # Extract dt from CFL calculation
                if "Computed dt:" in line:
                    current_dt = float(line.split(":")[-1].strip().split()[0])

                # Extract min edge length (dx approximation)
                if "Min element size:" in line:
                    current_dx = float(line.split(":")[-1].strip().split()[0])

                # Extract Newton residuals
                if "Newton iter" in line and "|Δx|" in line:
                    # Example: "  Newton iter 0: |Δx| = 1.234e-02, |Δx|/dt = 5.678e+00"
                    parts = line.split(",")
                    if len(parts) >= 2:
                        abs_res_str = parts[0].split("=")[-1].strip()
                        rel_res_str = parts[1].split("=")[-1].strip()

                        try:
                            abs_residual = float(abs_res_str)
                            rel_residual = float(rel_res_str)

                            newton_iter = int(line.split("Newton iter")[1].split(":")[0].strip())

                            if current_dt is not None and current_dx is not None:
                                residual_data.append({
                                    "profile": profile,
                                    "nx": nx,
                                    "cfl": cfl,
                                    "step": step_num,
                                    "newton_iter": newton_iter,
                                    "dt": current_dt,
                                    "dx": current_dx,
                                    "abs_residual": abs_residual,
                                    "rel_residual": rel_residual,
                                })
                        except ValueError:
                            pass

                # Track progress
                if "Progress:" in line and "step" in line:
                    try:
                        step_num = int(line.split("step")[1].split(",")[0].strip())
                    except:
                        pass

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return residual_data


def analyze_residuals(residual_data):
    """Analyze collected residual data and generate statistics."""

    if not residual_data:
        print("\nNo residual data collected!")
        return

    df = pd.DataFrame(residual_data)

    print("\n" + "="*80)
    print("RESIDUAL ANALYSIS RESULTS")
    print("="*80)

    # Statistics by profile
    print("\n--- Statistics by Profile ---")
    profile_stats = df.groupby("profile").agg({
        "abs_residual": ["min", "max", "mean", "median", "std"],
        "rel_residual": ["min", "max", "mean", "median", "std"],
        "dx": ["min", "max"],
        "dt": ["min", "max"],
    })
    print(profile_stats)

    # Statistics by nx (spatial resolution)
    print("\n--- Absolute Residual vs nx (spatial resolution) ---")
    nx_stats = df.groupby(["profile", "nx"]).agg({
        "abs_residual": ["min", "max", "mean", "median"],
        "dx": "mean",
    }).round(6)
    print(nx_stats)

    # Statistics by cfl (time step)
    print("\n--- Absolute Residual vs cfl (time step size) ---")
    cfl_stats = df.groupby(["profile", "cfl"]).agg({
        "abs_residual": ["min", "max", "mean", "median"],
        "dt": "mean",
    }).round(6)
    print(cfl_stats)

    # Scaling analysis
    print("\n--- Scaling Analysis ---")
    print("\nCorrelation between dx and abs_residual:")
    print(f"  {df['dx'].corr(df['abs_residual']):.4f}")

    print("\nCorrelation between dt and abs_residual:")
    print(f"  {df['dt'].corr(df['abs_residual']):.4f}")

    # Percentiles
    print("\n--- Absolute Residual Percentiles (all data) ---")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(df["abs_residual"], p)
        print(f"  {p:3d}th percentile: {val:.6e}")

    # Recommended absolute tolerances
    print("\n--- Recommended Absolute Tolerances (abs_tol) ---")
    print("\nBased on percentiles of |Δx| across all simulations:")
    print(f"  Very strict (1st percentile):   {np.percentile(df['abs_residual'], 1):.6e}")
    print(f"  Strict (10th percentile):       {np.percentile(df['abs_residual'], 10):.6e}")
    print(f"  Medium (50th percentile):       {np.percentile(df['abs_residual'], 50):.6e}")
    print(f"  Loose (90th percentile):        {np.percentile(df['abs_residual'], 90):.6e}")
    print(f"  Very loose (99th percentile):   {np.percentile(df['abs_residual'], 99):.6e}")

    # Save data
    output_path = "outputs/newton_residual_analysis.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_json(output_path, orient="records", indent=2)
    print(f"\n✅ Residual data saved to: {output_path}")

    # Create visualizations
    create_visualizations(df)

    return df


def create_visualizations(df):
    """Create visualization plots for residual analysis."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Newton Residual Analysis", fontsize=16)

    # Plot 1: Absolute residual vs dx by profile
    ax = axes[0, 0]
    for profile in df["profile"].unique():
        data = df[df["profile"] == profile]
        ax.scatter(data["dx"], data["abs_residual"], label=profile, alpha=0.6)
    ax.set_xlabel("dx (mesh spacing)")
    ax.set_ylabel("|Δx| (absolute residual)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Absolute Residual vs Mesh Spacing")

    # Plot 2: Absolute residual vs dt by profile
    ax = axes[0, 1]
    for profile in df["profile"].unique():
        data = df[df["profile"] == profile]
        ax.scatter(data["dt"], data["abs_residual"], label=profile, alpha=0.6)
    ax.set_xlabel("dt (time step)")
    ax.set_ylabel("|Δx| (absolute residual)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Absolute Residual vs Time Step")

    # Plot 3: Relative residual (current) vs absolute residual
    ax = axes[0, 2]
    ax.scatter(df["abs_residual"], df["rel_residual"], alpha=0.5)
    ax.set_xlabel("|Δx| (absolute residual)")
    ax.set_ylabel("|Δx|/dt (relative residual)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title("Current vs Proposed Residual")

    # Plot 4: Distribution of absolute residuals by profile
    ax = axes[1, 0]
    for profile in df["profile"].unique():
        data = df[df["profile"] == profile]["abs_residual"]
        ax.hist(np.log10(data), bins=20, alpha=0.5, label=profile)
    ax.set_xlabel("log10(|Δx|)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.set_title("Distribution of Absolute Residuals")

    # Plot 5: Absolute residual vs nx
    ax = axes[1, 1]
    for profile in df["profile"].unique():
        data = df.groupby(["profile", "nx"])["abs_residual"].median().loc[profile]
        ax.plot(data.index, data.values, marker='o', label=profile)
    ax.set_xlabel("nx (spatial resolution)")
    ax.set_ylabel("Median |Δx|")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Median Residual vs Spatial Resolution")

    # Plot 6: Absolute residual vs cfl
    ax = axes[1, 2]
    for profile in df["profile"].unique():
        data = df.groupby(["profile", "cfl"])["abs_residual"].median().loc[profile]
        ax.plot(data.index, data.values, marker='o', label=profile)
    ax.set_xlabel("CFL number")
    ax.set_ylabel("Median |Δx|")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Median Residual vs CFL Number")

    plt.tight_layout()

    output_path = "outputs/newton_residual_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Visualization saved to: {output_path}")
    plt.close()


def main():
    """Main analysis workflow."""

    print("Newton Residual Analysis for FEM2D Solver")
    print()
    print("Purpose: Understand how |Δx| scales with dx and dt")
    print("         to inform design of absolute tolerance (abs_tol)")
    print()
    print("Current: |Δx| < newton_v_res_tol * dt (velocity-like relative tolerance)")
    print("Proposed: |Δx| < abs_tol (absolute tolerance)")
    print()

    # Run simulations and collect residual data
    residual_data = run_with_residual_logging()

    if residual_data:
        # Analyze and visualize
        df = analyze_residuals(residual_data)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print()
        print("Next steps:")
        print("1. Review outputs/newton_residual_analysis.json for detailed data")
        print("2. Review outputs/newton_residual_analysis.png for visualizations")
        print("3. Use percentile-based recommendations to set abs_tol values")
        print("4. Consider profile-specific abs_tol if residuals vary significantly")
        print()
    else:
        print("\n❌ No residual data collected. Check simulation errors.")


if __name__ == "__main__":
    main()
