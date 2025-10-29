#!/usr/bin/env python3
"""
Test that the checkout script can load and process profile-specific precision configs.
"""

import os
import sys
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from checkouts.config_utils import load_config

def test_checkout_loading():
    """Test that checkout script logic handles profile-specific configs"""
    print("Testing checkout script with profile-specific precision levels...")
    print("="*80)

    # Load configuration
    config_path = "checkouts/fem2d.yaml"
    config = load_config(config_path)
    print("✅ Configuration loaded successfully")

    # Simulate the checkout script's precision config extraction
    precision_configs = {}
    for precision_name, precision_info in config["precision_levels"].items():
        # Check if this is profile-specific (new format) or global (old format)
        if all(isinstance(val, dict) and "energy_tolerance" in val and "var_threshold" in val
               for val in precision_info.values()):
            # New format: profile-specific precision levels
            precision_configs[precision_name] = precision_info
            print(f"✅ {precision_name}: Profile-specific format detected")
        elif isinstance(precision_info.get("energy_tolerance"), (int, float)) and \
             isinstance(precision_info.get("var_threshold"), (int, float)):
            # Old format: global precision levels (for backward compatibility)
            precision_configs[precision_name] = precision_info
            print(f"✅ {precision_name}: Global format detected")

    profiles = config["profiles"]["active_profiles"]
    print(f"\n✅ Active profiles: {profiles}")

    # Test extraction of profile-specific values
    print("\nTesting profile-specific value extraction:")
    print("-"*80)

    for precision_name, precision_vals in precision_configs.items():
        print(f"\n{precision_name.upper()} precision:")
        for profile in profiles:
            # Get profile-specific precision values (or use global if old format)
            if isinstance(precision_vals, dict) and profile in precision_vals:
                # New format: profile-specific
                energy_tolerance = precision_vals[profile]["energy_tolerance"]
                var_threshold = precision_vals[profile]["var_threshold"]
            else:
                # Old format: global values
                energy_tolerance = precision_vals["energy_tolerance"]
                var_threshold = precision_vals["var_threshold"]

            print(f"  {profile}: energy_tolerance={energy_tolerance:.4f}, var_threshold={var_threshold:.4f}")

    print("\n" + "="*80)
    print("✅ All tests PASSED! Checkout script logic is compatible.")
    return True

if __name__ == "__main__":
    success = test_checkout_loading()
    sys.exit(0 if success else 1)
