#!/usr/bin/env python3
"""
Analyze existing FEM2D simulation folders to determine wall times from VTK file timestamps.

This script:
1. Scans all FEM2D simulation directories
2. Finds first and last VTK files by creation time
3. Calculates wall_time_total from timestamp difference
4. Determines if wall_time_exceeded (using max_wall_time=120s)
5. Reports statistics and maximum observed wall time

Usage:
    python scripts/analyze_fem2d_wall_times.py [--sim-dir PATH] [--max-wall-time SECONDS]
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def find_vtk_files(sim_dir: str) -> List[str]:
    """Find all VTK files in a simulation directory."""
    vtk_pattern = os.path.join(sim_dir, "frame_*.vtk")
    vtk_files = glob.glob(vtk_pattern)
    return sorted(vtk_files)


def get_wall_time_from_vtk(sim_dir: str) -> Optional[float]:
    """
    Calculate wall time from VTK file creation timestamps.

    Returns wall time in seconds, or None if calculation failed.
    """
    vtk_files = find_vtk_files(sim_dir)

    if len(vtk_files) < 2:
        print(f"  ⚠️  Warning: Only {len(vtk_files)} VTK file(s) found, need at least 2")
        return None

    # Get creation times for first and last files
    first_file = vtk_files[0]
    last_file = vtk_files[-1]

    try:
        first_time = os.path.getctime(first_file)
        last_time = os.path.getctime(last_file)
        wall_time = last_time - first_time

        return wall_time
    except Exception as e:
        print(f"  ❌ Error calculating wall time: {e}")
        return None


def analyze_simulation_folder(sim_dir: str, max_wall_time: float) -> Dict:
    """
    Analyze a single simulation folder.

    Returns dictionary with analysis results.
    """
    result = {
        "path": sim_dir,
        "name": os.path.basename(sim_dir),
        "wall_time_total": None,
        "wall_time_exceeded": None,
        "has_meta": False,
        "num_vtk_files": 0,
    }

    # Check for meta.json
    meta_path = os.path.join(sim_dir, "meta.json")
    if os.path.exists(meta_path):
        result["has_meta"] = True
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                # Check if wall_time fields already exist
                if "wall_time_total" in meta:
                    print(f"  ℹ️  meta.json already has wall_time_total: {meta['wall_time_total']:.2f}s")
                    result["wall_time_total"] = meta["wall_time_total"]
                    result["wall_time_exceeded"] = meta.get("wall_time_exceeded", False)
                    return result
        except Exception as e:
            print(f"  ⚠️  Warning: Could not read meta.json: {e}")

    # Calculate from VTK timestamps
    vtk_files = find_vtk_files(sim_dir)
    result["num_vtk_files"] = len(vtk_files)

    if len(vtk_files) < 2:
        print(f"  ⚠️  Insufficient VTK files ({len(vtk_files)}), skipping")
        return result

    wall_time = get_wall_time_from_vtk(sim_dir)

    if wall_time is not None:
        result["wall_time_total"] = wall_time
        result["wall_time_exceeded"] = wall_time > max_wall_time
        print(f"  ✅ Wall time: {wall_time:.2f}s, Exceeded: {result['wall_time_exceeded']}")

    return result


def find_fem2d_sim_directories(base_dir: str) -> List[str]:
    """Find all FEM2D simulation directories."""
    # Look for directories matching typical FEM2D pattern
    # e.g., sim_res/fem2d/p1_dx0.1_cfl0.5/
    pattern = os.path.join(base_dir, "sim_res", "fem2d", "*")

    dirs = []
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            # Check if it has VTK files
            vtk_files = find_vtk_files(path)
            if vtk_files:
                dirs.append(path)

    return sorted(dirs)


def print_summary(results: List[Dict], max_wall_time: float):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total = len(results)
    valid = [r for r in results if r["wall_time_total"] is not None]
    exceeded = [r for r in valid if r["wall_time_exceeded"]]

    print(f"Total directories analyzed: {total}")
    print(f"Valid wall time calculations: {len(valid)}")
    print(f"Simulations exceeding {max_wall_time}s: {len(exceeded)}")

    if valid:
        wall_times = [r["wall_time_total"] for r in valid]
        max_wt = max(wall_times)
        min_wt = min(wall_times)
        avg_wt = sum(wall_times) / len(wall_times)

        print(f"\nWall Time Statistics:")
        print(f"  Min: {min_wt:.2f}s")
        print(f"  Max: {max_wt:.2f}s")
        print(f"  Avg: {avg_wt:.2f}s")
        print(f"\n⚡ Recommended max_wall_time: {max_wt:.0f}s (rounded up from {max_wt:.2f}s)")

    if exceeded:
        print(f"\n⚠️  Directories exceeding {max_wall_time}s:")
        for r in exceeded:
            print(f"  - {r['name']}: {r['wall_time_total']:.2f}s")

    print("\n" + "="*80)


def save_results(results: List[Dict], output_file: str):
    """Save analysis results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FEM2D simulation wall times from VTK file timestamps"
    )
    parser.add_argument(
        "--sim-dir",
        type=str,
        default=".",
        help="Base directory containing sim_res/fem2d/ (default: current directory)"
    )
    parser.add_argument(
        "--max-wall-time",
        type=float,
        default=120.0,
        help="Maximum wall time threshold in seconds (default: 120)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fem2d_wall_time_analysis.json",
        help="Output JSON file for results (default: fem2d_wall_time_analysis.json)"
    )

    args = parser.parse_args()

    print("🔍 Analyzing FEM2D simulation wall times...")
    print(f"Base directory: {os.path.abspath(args.sim_dir)}")
    print(f"Max wall time threshold: {args.max_wall_time}s")
    print("="*80)

    # Find all simulation directories
    sim_dirs = find_fem2d_sim_directories(args.sim_dir)

    if not sim_dirs:
        print("❌ No FEM2D simulation directories found!")
        print(f"   Searched in: {os.path.join(args.sim_dir, 'sim_res', 'fem2d')}")
        sys.exit(1)

    print(f"Found {len(sim_dirs)} simulation directories\n")

    # Analyze each directory
    results = []
    for sim_dir in sim_dirs:
        print(f"Analyzing: {os.path.basename(sim_dir)}")
        result = analyze_simulation_folder(sim_dir, args.max_wall_time)
        results.append(result)

    # Print summary
    print_summary(results, args.max_wall_time)

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
