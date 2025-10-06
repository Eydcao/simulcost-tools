#!/usr/bin/env python3
"""
Test script for CG residual tracking functionality in Hasegawa-Mima solver
"""

import sys
import os
import json

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from runners.hasegawa_mima_linear import main as run_simulation

def test_residual_tracking():
    """Test the CG residual tracking functionality"""
    print("Testing CG residual tracking functionality...")

    # Test with different cg_atol values to see sensitivity
    test_configs = [
        {"cg_atol": 1e-4, "name": "loose"},
        {"cg_atol": 1e-6, "name": "medium"},
        {"cg_atol": 1e-8, "name": "tight"}
    ]

    for config in test_configs:
        print(f"\n--- Testing with cg_atol = {config['cg_atol']:.1e} ({config['name']}) ---")

        # Run simulation with small parameters for fast testing
        os.system(f"""python runners/hasegawa_mima_linear.py --config-name=p1 \
                     N=64 dt=20.0 cg_atol={config['cg_atol']:.1e} \
                     record_dt=100 end_frame=3 verbose=true \
                     dump_dir="sim_res/hasegawa_mima_linear/test_residual_{config['name']}" """)

        # Check if residual analysis was generated
        sim_dir = f"sim_res/hasegawa_mima_linear/test_residual_{config['name']}_N_64_dt_2.00e+01_cg_{config['cg_atol']:.2e}_numerical"

        meta_file = os.path.join(sim_dir, "meta.json")
        residual_plot = os.path.join(sim_dir, "cg_residual_analysis.png")
        residual_summary = os.path.join(sim_dir, "cg_residual_summary.txt")

        if os.path.exists(meta_file):
            print(f"✓ Simulation completed: {sim_dir}")

            # Check metadata
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if 'cg_residual_trajectories' in meta and meta['cg_residual_trajectories']:
                    print(f"✓ Found {len(meta['cg_residual_trajectories'])} CG residual trajectories")

                    # Show first trajectory info
                    first_traj = meta['cg_residual_trajectories'][0]
                    print(f"  First trajectory: Step {first_traj['step']}, {first_traj['iterations']} iterations")
                    if first_traj['residual_trajectory']:
                        initial_res = first_traj['residual_trajectory'][0]
                        final_res = first_traj['residual_trajectory'][-1]
                        print(f"  Residual: {initial_res:.2e} → {final_res:.2e} (reduction: {initial_res/final_res:.1e})")
                else:
                    print("✗ No residual trajectories found in metadata")

            # Check analysis files
            if os.path.exists(residual_plot):
                print(f"✓ Residual analysis plot created")
            else:
                print(f"✗ Residual analysis plot missing")

            if os.path.exists(residual_summary):
                print(f"✓ Residual summary file created")
                # Show first few lines
                with open(residual_summary, 'r') as f:
                    lines = f.readlines()[:10]
                    print("  Summary preview:")
                    for line in lines:
                        print(f"    {line.rstrip()}")
            else:
                print(f"✗ Residual summary file missing")
        else:
            print(f"✗ Simulation failed or metadata missing")

    print(f"\n=== Residual Tracking Test Complete ===")
    print(f"Check the generated analysis files in sim_res/hasegawa_mima_linear/")

if __name__ == "__main__":
    test_residual_tracking()