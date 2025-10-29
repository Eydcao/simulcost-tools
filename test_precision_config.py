#!/usr/bin/env python3
"""
Quick test script to verify the profile-specific precision configuration loads correctly.
"""

import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """Test that the configuration loads and parses correctly"""
    print("Testing FEM2D profile-specific precision configuration...")
    print("="*80)

    # Load config
    config_path = "checkouts/fem2d.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Test precision levels
    print("\n1. Testing Precision Levels Structure:")
    precision_levels = config["precision_levels"]

    for precision_name in ["high", "medium", "low"]:
        print(f"\n{precision_name.upper()} precision:")
        precision_data = precision_levels[precision_name]

        # Check that all profiles are present
        for profile in ["p1", "p2", "p3"]:
            if profile not in precision_data:
                print(f"  ❌ ERROR: Profile {profile} not found!")
                return False

            energy_tol = precision_data[profile]["energy_tolerance"]
            var_thr = precision_data[profile]["var_threshold"]

            print(f"  {profile}: energy_tolerance={energy_tol:.3f}, var_threshold={var_thr:.3f}")

    # Test profiles
    print("\n2. Testing Active Profiles:")
    profiles = config["profiles"]["active_profiles"]
    print(f"  Active profiles: {profiles}")

    # Test target parameters
    print("\n3. Testing Target Parameters:")
    target_params = config["target_parameters"]
    for param_name in target_params:
        print(f"  - {param_name}")
        param_config = target_params[param_name]
        print(f"    Initial values: {param_config['initial_values']}")

    print("\n" + "="*80)
    print("✅ Configuration test PASSED!")
    print("\nSummary of precision thresholds:")
    print("\n{:<10} {:<8} {:<18} {:<18}".format("Precision", "Profile", "energy_tolerance", "var_threshold"))
    print("-"*80)

    for precision_name in ["high", "medium", "low"]:
        for profile in ["p1", "p2", "p3"]:
            energy_tol = precision_levels[precision_name][profile]["energy_tolerance"]
            var_thr = precision_levels[precision_name][profile]["var_threshold"]
            print("{:<10} {:<8} {:<18.4f} {:<18.4f}".format(precision_name, profile, energy_tol, var_thr))

    return True

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)
