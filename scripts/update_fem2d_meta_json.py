#!/usr/bin/env python3
"""
Update existing FEM2D meta.json files to add wall_time_total and wall_time_exceeded fields.

This script updates meta.json files IN PLACE without renaming folders.
All existing simulations are treated as unconstrained reference runs.

Usage:
    python scripts/update_fem2d_meta_json.py [--analysis-file FILE] [--dry-run]
"""

import os
import sys
import json
import argparse
from typing import Dict, List


def update_single_meta_json(sim_dir: str, wall_time_total: float, dry_run: bool = False) -> bool:
    """
    Update meta.json in a simulation directory.

    Args:
        sim_dir: Path to simulation directory
        wall_time_total: Wall time in seconds
        dry_run: If True, don't actually write changes

    Returns:
        True if successful, False otherwise
    """
    meta_path = os.path.join(sim_dir, "meta.json")

    if not os.path.exists(meta_path):
        print(f"  ⚠️  Warning: meta.json not found at {meta_path}")
        return False

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Check if already updated
        if "wall_time_total" in meta and "wall_time_exceeded" in meta:
            print(f"  ℹ️  Already has wall_time fields, skipping")
            return True

        # Add wall time fields (all existing runs are unconstrained references)
        meta["wall_time_total"] = float(wall_time_total)
        meta["wall_time_exceeded"] = False  # All existing runs completed (unconstrained)

        if not dry_run:
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=4)
            print(f"  ✅ Updated: wall_time_total={wall_time_total:.2f}s, exceeded=False")
        else:
            print(f"  [DRY RUN] Would update: wall_time_total={wall_time_total:.2f}s, exceeded=False")

        return True

    except Exception as e:
        print(f"  ❌ Error updating meta.json: {e}")
        return False


def load_analysis_results(analysis_file: str) -> List[Dict]:
    """Load analysis results from JSON file."""
    if not os.path.exists(analysis_file):
        print(f"❌ Analysis file not found: {analysis_file}")
        print(f"   Please run analyze_fem2d_wall_times.py first!")
        sys.exit(1)

    with open(analysis_file, 'r') as f:
        results = json.load(f)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Update FEM2D meta.json files with wall_time fields"
    )
    parser.add_argument(
        "--analysis-file",
        type=str,
        default="fem2d_wall_time_analysis.json",
        help="Input JSON file from analyze_fem2d_wall_times.py"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually updating files"
    )

    args = parser.parse_args()

    mode = "[DRY RUN MODE] " if args.dry_run else ""
    print(f"🔧 {mode}Updating FEM2D meta.json files...")
    print(f"Analysis file: {args.analysis_file}")
    print("="*80)

    # Load analysis results
    results = load_analysis_results(args.analysis_file)
    print(f"Loaded {len(results)} simulation folders from analysis\n")

    # Update each meta.json
    success_count = 0
    skip_count = 0
    fail_count = 0

    for result in results:
        sim_dir = result["path"]
        wall_time_total = result.get("wall_time_total")

        print(f"Processing: {os.path.basename(sim_dir)}")

        if wall_time_total is None:
            print(f"  ⚠️  No wall_time_total available, skipping")
            skip_count += 1
            continue

        success = update_single_meta_json(sim_dir, wall_time_total, args.dry_run)

        if success:
            success_count += 1
        else:
            fail_count += 1

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{mode}Successfully updated: {success_count}")
    print(f"{mode}Skipped: {skip_count}")
    print(f"{mode}Failed: {fail_count}")

    if args.dry_run:
        print("\n⚠️  This was a DRY RUN - no actual changes were made!")
        print("   Remove --dry-run flag to apply changes.")

    print("="*80)


if __name__ == "__main__":
    main()
