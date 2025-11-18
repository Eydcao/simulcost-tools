#!/usr/bin/env python3
"""
Delete FEM2D simulation folders with wall_time suffix.

This script removes incorrectly generated simulation folders that have the
_wall_time_{number} suffix. These were created with the wrong case name
and need to be regenerated.

Usage:
    # Preview all wall_time folders
    python scripts/delete_wall_time_folders.py --dry-run

    # Preview only p1 wall_time folders
    python scripts/delete_wall_time_folders.py --dry-run --profile p1

    # Delete only p1 wall_time folders
    python scripts/delete_wall_time_folders.py --profile p1

    # Delete all wall_time folders
    python scripts/delete_wall_time_folders.py
"""

import os
import re
import shutil
import argparse
from pathlib import Path


def find_wall_time_folders(base_dir="sim_res/fem2d", profile_filter=None):
    """
    Find all directories matching pattern: {profile}_dx{value}_cfl{value}_wall_time_{number}

    Args:
        base_dir: Base directory containing simulation results
        profile_filter: Optional profile name to filter (e.g., "p1", "p2")

    Returns:
        List of Path objects for matching directories
    """
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return []

    # Pattern: p{N}_dx{value}_cfl{value}_wall_time_{number}
    # Examples:
    # - p1_dx0.5_cfl0.5_wall_time_120
    # - p2_dx0.25_cfl1.0_wall_time_120
    # - p3_dx0.125_cfl2.0_wall_time_120
    if profile_filter:
        # Filter by specific profile
        pattern = re.compile(rf'^{re.escape(profile_filter)}_dx[\d.]+_cfl[\d.]+_wall_time_\d+$')
    else:
        # Match all profiles
        pattern = re.compile(r'^p\d+_dx[\d.]+_cfl[\d.]+_wall_time_\d+$')

    matching_folders = []
    base_path = Path(base_dir)

    for item in base_path.iterdir():
        if item.is_dir() and pattern.match(item.name):
            matching_folders.append(item)

    return sorted(matching_folders)


def get_folder_info(folder_path):
    """
    Get information about a folder (size, file count).

    Args:
        folder_path: Path object for the folder

    Returns:
        Dictionary with folder info
    """
    total_size = 0
    file_count = 0

    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
                    file_count += 1
    except Exception as e:
        print(f"Warning: Could not get info for {folder_path}: {e}")

    return {
        "size_bytes": total_size,
        "size_mb": total_size / (1024 * 1024),
        "file_count": file_count
    }


def format_size(size_mb):
    """Format size in MB or GB."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"
    else:
        return f"{size_mb:.2f} MB"


def main():
    parser = argparse.ArgumentParser(
        description="Delete FEM2D simulation folders with wall_time suffix"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--base-dir",
        default="sim_res/fem2d",
        help="Base directory containing simulation results (default: sim_res/fem2d)"
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Filter by specific profile (e.g., p1, p2, p3). If not specified, all profiles are included."
    )
    args = parser.parse_args()

    print("=" * 80)
    print("FEM2D Wall Time Folder Deletion Tool")
    print("=" * 80)
    print()

    # Find matching folders
    print(f"Searching in: {args.base_dir}")
    if args.profile:
        print(f"Profile filter: {args.profile}")
    folders = find_wall_time_folders(args.base_dir, profile_filter=args.profile)

    if not folders:
        print("No folders with _wall_time suffix found.")
        return

    print(f"\nFound {len(folders)} folders to delete:")
    print()

    # Calculate total size
    total_size_mb = 0
    total_files = 0

    for folder in folders:
        info = get_folder_info(folder)
        total_size_mb += info["size_mb"]
        total_files += info["file_count"]

        print(f"  {folder.name}")
        print(f"    Size: {format_size(info['size_mb'])}, Files: {info['file_count']}")

    print()
    print(f"Total: {len(folders)} folders, {format_size(total_size_mb)}, {total_files} files")
    print()

    if args.dry_run:
        print("DRY RUN - Nothing deleted (use without --dry-run to actually delete)")
        return

    # Ask for confirmation
    response = input("Are you sure you want to delete these folders? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled.")
        return

    # Delete folders
    print()
    print("Deleting folders...")
    deleted_count = 0
    failed_count = 0

    for folder in folders:
        try:
            shutil.rmtree(folder)
            print(f"  ✓ Deleted: {folder.name}")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ Failed to delete {folder.name}: {e}")
            failed_count += 1

    print()
    print(f"Deletion complete:")
    print(f"  Successfully deleted: {deleted_count} folders")
    if failed_count > 0:
        print(f"  Failed: {failed_count} folders")
    print(f"  Freed space: ~{format_size(total_size_mb)}")


if __name__ == "__main__":
    main()
