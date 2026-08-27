#!/usr/bin/env python3
"""
Fix incorrect header metadata in Crab pulsar .fil files.

Corrects:
  - src_raj / src_dej to the true J0534+2200 coordinates
    (previous edit accidentally wrote 21:50:52.061 instead of 22:00:52.061)
  - az_start / za_start, which are stale/incorrect from the original
    recording and are not recomputed automatically when RA/Dec change.

This edits files IN PLACE (header bytes only, data untouched). No backup
is made automatically -- see the --dry-run and --backup options below.

Usage:
    python fix_crab_headers.py /path/to/dir_with_fil_files
    python fix_crab_headers.py /path/to/dir --dry-run
    python fix_crab_headers.py /path/to/dir --backup
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from sigpyproc.io import sigproc

# --- Correct values for J0534+2200 (Crab pulsar) ---------------------------
SOURCE_NAME = "J0534+2200"
SRC_RAJ = 53431       # 05:34:31.973 in sigproc hhmmss.ss format
SRC_DEJ = 220052      # 22:00:52.061 in sigproc ddmmss.ss format (NOTE: not 21:50:52)
AZ_START = 0.0
ZA_START = 0.0

FIELDS = {
    "source_name": SOURCE_NAME,
    "src_raj": SRC_RAJ,
    "src_dej": SRC_DEJ,
    "az_start": AZ_START,
    "za_start": ZA_START,
}


def fix_file(fil_path: Path, *, dry_run: bool = False, backup: bool = False) -> None:
    print(f"\n{fil_path}")

    if backup and not dry_run:
        backup_path = fil_path.with_suffix(fil_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(fil_path, backup_path)
            print(f"  backed up -> {backup_path.name}")
        else:
            print(f"  backup already exists, skipping -> {backup_path.name}")

    for key, value in FIELDS.items():
        if dry_run:
            print(f"  [dry-run] would set {key} = {value}")
            continue
        try:
            sigproc.edit_header(fil_path, key, value)
            print(f"  set {key} = {value}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to set {key} on {fil_path}: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing .fil files to fix (searched recursively)",
    )
    parser.add_argument(
        "--pattern",
        default="*.fil",
        help="Glob pattern for files to match (default: *.fil)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directory recursively (uses rglob instead of glob)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without writing anything",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy each file to <file>.fil.bak before editing",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"Not a directory: {args.directory}")

    glob_fn = args.directory.rglob if args.recursive else args.directory.glob
    fil_files = sorted(glob_fn(args.pattern))

    if not fil_files:
        print(f"No files matching '{args.pattern}' found in {args.directory}")
        return

    print(f"Found {len(fil_files)} file(s) to update:")
    for f in fil_files:
        print(f"  {f}")

    if args.dry_run:
        print("\n--- DRY RUN: no files will be modified ---")

    for f in fil_files:
        fix_file(f, dry_run=args.dry_run, backup=args.backup)

    print("\nDone.")


if __name__ == "__main__":
    main()
