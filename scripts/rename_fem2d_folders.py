#!/usr/bin/env python3
"""
Rename FEM2D simulation folders to include wall_time suffix for cache disambiguation.

This script:
1. Reads analysis results from analyze_fem2d_wall_times.py
2. Renames folders to include _wall_time_{max_wall_time} suffix
3. Updates meta.json with wall_time_total and wall_time_exceeded fields

Usage:
    # Step 1: Analyze existing folders
    python scripts/analyze_fem2d_wall_times.py

    # Step 2: After reviewing max time, rename folders
    python scripts/rename_fem2d_folders.py --max-wall-time 150

    # Dry run (preview changes without applying)
    python scripts/rename_fem2d_folders.py --max-wall-time 150 --dry-run
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def update_meta_json(meta_path: str, wall_time_total: float, wall_time_exceeded: bool, dry_run: bool = False):
    """
    Update meta.json with wall time fields.
    """
    if not os.path.exists(meta_path):
        print(f"    ⚠️  Warning: meta.json not found at {meta_path}")
        return

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Add wall time fields
        meta["wall_time_total"] = float(wall_time_total)
        meta["wall_time_exceeded"] = bool(wall_time_exceeded)

        if not dry_run:
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=4)
            print(f"    ✅ Updated meta.json: wall_time_total={wall_time_total:.2f}s, exceeded={wall_time_exceeded}")
        else:
            print(f"    [DRY RUN] Would update meta.json: wall_time_total={wall_time_total:.2f}s, exceeded={wall_time_exceeded}")

    except Exception as e:
        print(f"    ❌ Error updating meta.json: {e}")


def rename_folder_with_suffix(old_path: str, max_wall_time: float, dry_run: bool = False) -> Optional[str]:
    """
    Rename folder to include wall_time suffix.

    Returns new path if successful, None otherwise.
    """
    # Check if folder already has wall_time suffix
    if "_wall_time_" in os.path.basename(old_path):
        print(f"    ℹ️  Already has wall_time suffix, skipping")
        return old_path

    # Construct new path with suffix
    new_path = old_path + f"_wall_time_{int(max_wall_time)}"

    # Check if target already exists
    if os.path.exists(new_path):
        print(f"    ⚠️  Warning: Target already exists: {new_path}")
        return None

    if not dry_run:
        try:
            shutil.move(old_path, new_path)
            print(f"    ✅ Renamed to: {os.path.basename(new_path)}")
            return new_path
        except Exception as e:
            print(f"    ❌ Error renaming: {e}")
            return None
    else:
        print(f"    [DRY RUN] Would rename to: {os.path.basename(new_path)}")
        return new_path


def process_simulation_folder(result: Dict, max_wall_time: float, dry_run: bool = False) -> Dict:
    """
    Process a single simulation folder:
    1. Rename folder with wall_time suffix
    2. Update meta.json with wall_time fields

    Returns updated result dictionary.
    """
    old_path = result["path"]
    wall_time_total = result.get("wall_time_total")

    if wall_time_total is None:
        print(f"  ⚠️  No wall_time_total available, skipping")
        return result

    wall_time_exceeded = wall_time_total > max_wall_time

    print(f"\nProcessing: {os.path.basename(old_path)}")
    print(f"  Wall time: {wall_time_total:.2f}s, Exceeded: {wall_time_exceeded}")

    # Rename folder
    new_path = rename_folder_with_suffix(old_path, max_wall_time, dry_run)

    if new_path is not None:
        # Update meta.json in new location
        meta_path = os.path.join(new_path, "meta.json")
        update_meta_json(meta_path, wall_time_total, wall_time_exceeded, dry_run)

        # Update result dictionary
        result["new_path"] = new_path
        result["renamed"] = True
        result["wall_time_exceeded"] = wall_time_exceeded
    else:
        result["renamed"] = False

    return result


def load_analysis_results(analysis_file: str) -> List[Dict]:
    """Load analysis results from JSON file."""
    if not os.path.exists(analysis_file):
        print(f"❌ Analysis file not found: {analysis_file}")
        print(f"   Please run analyze_fem2d_wall_times.py first!")
        sys.exit(1)

    with open(analysis_file, 'r') as f:
        results = json.load(f)

    return results


def print_summary(results: List[Dict], dry_run: bool):
    """Print summary of operations."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    renamed = [r for r in results if r.get("renamed", False)]
    skipped = [r for r in results if not r.get("renamed", False)]

    mode = "[DRY RUN] " if dry_run else ""
    print(f"{mode}Folders processed: {len(results)}")
    print(f"{mode}Successfully renamed: {len(renamed)}")
    print(f"{mode}Skipped: {len(skipped)}")

    if dry_run:
        print("\n⚠️  This was a DRY RUN - no actual changes were made!")
        print("   Remove --dry-run flag to apply changes.")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Rename FEM2D simulation folders with wall_time suffix"
    )
    parser.add_argument(
        "--max-wall-time",
        type=float,
        required=True,
        help="Maximum wall time value for suffix (e.g., 150)"
    )
    parser.add_argument(
        "--analysis-file",
        type=str,
        default="fem2d_wall_time_analysis.json",
        help="Input JSON file from analyze_fem2d_wall_times.py (default: fem2d_wall_time_analysis.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually renaming folders"
    )

    args = parser.parse_args()

    mode = "[DRY RUN MODE] " if args.dry_run else ""
    print(f"🔧 {mode}Renaming FEM2D simulation folders...")
    print(f"Max wall time: {args.max_wall_time}s")
    print(f"Analysis file: {args.analysis_file}")
    print("="*80)

    # Load analysis results
    results = load_analysis_results(args.analysis_file)
    print(f"Loaded {len(results)} simulation folders from analysis\n")

    # Process each folder
    updated_results = []
    for result in results:
        updated_result = process_simulation_folder(result, args.max_wall_time, args.dry_run)
        updated_results.append(updated_result)

    # Print summary
    print_summary(updated_results, args.dry_run)

    # Save updated results
    if not args.dry_run:
        output_file = "fem2d_rename_results.json"
        with open(output_file, 'w') as f:
            json.dump(updated_results, f, indent=2)
        print(f"\n💾 Updated results saved to: {output_file}")


if __name__ == "__main__":
    main()
