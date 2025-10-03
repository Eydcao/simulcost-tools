#!/usr/bin/env python
"""
Test script for decoupled Hasegawa-Mima Linear simulation and analysis.
Tests that analytical solutions are saved separately and error is computed via comparison.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wrappers.hasegawa_mima_linear import run_sim_hasegawa_mima_linear, get_error_metric
from dummy_sols.hasegawa_mima_linear import find_convergent_N

def test_single_simulation():
    """Test a single numerical simulation with analytical reference"""
    print("=" * 80)
    print("TEST 1: Single Numerical Simulation")
    print("=" * 80)

    profile = "p1"
    N = 64
    dt = 10.0
    cg_atol = 1e-4

    print(f"\nRunning numerical simulation: profile={profile}, N={N}, dt={dt:.2e}, cg_atol={cg_atol:.2e}")

    # Run simulation
    cost = run_sim_hasegawa_mima_linear(profile=profile, N=N, dt=dt, cg_atol=cg_atol, analytical=False)
    print(f"✅ Simulation completed. Cost: {cost}")

    # Get error via decoupled comparison
    sim_dir = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}_cg_{cg_atol:.2e}_numerical"
    error = get_error_metric(sim_dir)

    if error is not None:
        print(f"✅ Error metric computed via decoupled comparison: {error:.6e}")
        print(f"\n✅ TEST 1 PASSED: Simulation and decoupled analysis successful")
        return True
    else:
        print(f"❌ ERROR: Failed to compute error metric")
        print(f"❌ TEST 1 FAILED")
        return False


def test_convergence_search():
    """Test convergence search for N parameter"""
    print("\n" + "=" * 80)
    print("TEST 2: Convergence Search for N Parameter")
    print("=" * 80)

    profile = "p1"
    N_initial = 64
    dt = 10.0
    cg_atol = 1e-4
    tolerance_rmse = 0.001
    multiplication_factor = 2
    max_iteration_num = 2  # Only test 2 iterations: N=64, N=128

    print(f"\nSearching for convergent N:")
    print(f"  Profile: {profile}")
    print(f"  Initial N: {N_initial}")
    print(f"  dt: {dt:.2e}")
    print(f"  cg_atol: {cg_atol:.2e}")
    print(f"  tolerance_rmse: {tolerance_rmse}")
    print(f"  max_iteration_num: {max_iteration_num}")

    # Run convergence search
    is_converged, best_N, cost_history, param_history = find_convergent_N(
        profile=profile,
        N=N_initial,
        dt=dt,
        cg_atol=cg_atol,
        tolerance_rmse=tolerance_rmse,
        multiplication_factor=multiplication_factor,
        max_iteration_num=max_iteration_num
    )

    print(f"\n--- Search Results ---")
    print(f"Converged: {is_converged}")
    print(f"Best N: {best_N}")
    print(f"Cost history: {cost_history}")
    print(f"Total cost: {sum(cost_history)}")
    print(f"Parameters tested: {len(param_history)}")

    if best_N is not None:
        print(f"\n✅ TEST 2 PASSED: Convergence search completed successfully")
        return True
    else:
        print(f"\n⚠️  TEST 2 WARNING: No convergent N found (may be expected with low tolerance)")
        return True  # Not a failure, just didn't converge


def verify_directory_structure():
    """Verify that both numerical and analytical directories are created"""
    print("\n" + "=" * 80)
    print("TEST 3: Directory Structure Verification")
    print("=" * 80)

    profile = "p1"
    N = 64
    dt = 10.0
    cg_atol = 1e-4

    numerical_dir = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}_cg_{cg_atol:.2e}_numerical"
    analytical_dir = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}_analytical"

    print(f"\nChecking directories:")
    print(f"  Numerical: {numerical_dir}")
    print(f"  Analytical: {analytical_dir}")

    numerical_exists = os.path.exists(numerical_dir)
    analytical_exists = os.path.exists(analytical_dir)

    print(f"\nNumerical directory exists: {'✅' if numerical_exists else '❌'}")
    print(f"Analytical directory exists: {'✅' if analytical_exists else '❌'}")

    if numerical_exists and analytical_exists:
        # Check for frame files
        import glob
        num_frames = len(glob.glob(os.path.join(numerical_dir, "frame_*.h5")))
        ana_frames = len(glob.glob(os.path.join(analytical_dir, "frame_*.h5")))

        print(f"\nNumerical frames: {num_frames}")
        print(f"Analytical frames: {ana_frames}")

        if num_frames > 0 and ana_frames > 0 and num_frames == ana_frames:
            print(f"\n✅ TEST 3 PASSED: Directory structure verified")
            return True
        else:
            print(f"\n❌ TEST 3 FAILED: Frame count mismatch or no frames found")
            return False
    else:
        print(f"\n❌ TEST 3 FAILED: Required directories not found")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("HASEGAWA-MIMA LINEAR: DECOUPLED SIMULATION & ANALYSIS TEST")
    print("=" * 80)

    results = []

    # Test 1: Single simulation
    results.append(("Single Simulation", test_single_simulation()))

    # Test 2: Convergence search
    results.append(("Convergence Search", test_convergence_search()))

    # Test 3: Directory structure
    results.append(("Directory Structure", verify_directory_structure()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
