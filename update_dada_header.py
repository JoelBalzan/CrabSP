#!/usr/bin/env python3
"""
Fix a mislabeled SOURCE/RA/DEC in raw PSRDADA (.dada) ASCII headers.

Background: some .dada files in the Crab (J0534+2200) dataset were written
with the DADA header still carrying the coordinates/source name of the
PREVIOUS scheduled pointing (Vela, J0835-4510), even though the telescope
was actually tracking the Crab pulsar for this observation. This patches
SOURCE / RA / DEC in place to the correct Crab values.

The DADA header is a fixed-size (HDR_SIZE bytes, taken from the header
itself) null-padded ASCII block at the start of the file, followed
immediately by the raw baseband data. This script only ever touches bytes
within that fixed-size header block -- the data payload is never read or
moved.

Usage:
    python fix_dada_headers.py /path/to/dir --dry-run
    python fix_dada_headers.py /path/to/dir --backup
    python fix_dada_headers.py /path/to/dir --pattern "*.dada" --recursive
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# --- Correct values for J0534+2200 (Crab pulsar) ---------------------------
SOURCE = "J0534+2200"
RA = "05:34:31"
DEC = "+22:00:52"

# Key: (regex to match the existing "KEY<spaces>value" line, new value)
FIELD_PATTERNS = {
    "SOURCE": (re.compile(r"^(SOURCE\s+)\S+", re.MULTILINE), SOURCE),
    "RA": (re.compile(r"^(RA\s+)\S+", re.MULTILINE), RA),
    "DEC": (re.compile(r"^(DEC\s+)\S+", re.MULTILINE), DEC),
}

HDR_SIZE_RE = re.compile(r"^HDR_SIZE\s+(\d+)", re.MULTILINE)


def read_header(path: Path) -> tuple[str, int]:
    """Read enough of the file to find HDR_SIZE, then return the decoded
    header text (nulls stripped) and the declared HDR_SIZE in bytes."""
    with path.open("rb") as fp:
        # HDR_SIZE is always near the top; 4096 bytes is enough to find it
        # for any sane DADA header. Read a bit more to be safe.
        probe = fp.read(8192)

    probe_text = probe.decode("ascii", errors="replace")
    m = HDR_SIZE_RE.search(probe_text)
    if not m:
        msg = f"Could not find HDR_SIZE in header of {path}"
        raise ValueError(msg)
    hdr_size = int(m.group(1))

    with path.open("rb") as fp:
        raw = fp.read(hdr_size)

    text = raw.decode("ascii", errors="replace").rstrip("\x00")
    return text, hdr_size


def patch_header(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """Apply the field substitutions, returning new text and a log of
    (old_value, new_value) per key that was actually changed."""
    changes: dict[str, tuple[str, str]] = {}
    new_text = text
    for key, (pattern, new_value) in FIELD_PATTERNS.items():
        match = pattern.search(new_text)
        if not match:
            print(f"  WARNING: key '{key}' not found in header, skipping", file=sys.stderr)
            continue
        old_value = match.group(0)[len(match.group(1)):]
        replacement = match.group(1) + new_value
        new_text = new_text[: match.start()] + replacement + new_text[match.end():]
        changes[key] = (old_value, new_value)
    return new_text, changes


def fix_file(path: Path, *, dry_run: bool = False, backup: bool = False) -> None:
    print(f"\n{path}")

    text, hdr_size = read_header(path)
    new_text, changes = patch_header(text)

    if not changes:
        print("  no matching fields found, nothing to do")
        return

    for key, (old, new) in changes.items():
        print(f"  {key}: {old!r} -> {new!r}")

    if len(new_text.encode("ascii")) > hdr_size:
        msg = (
            f"Patched header ({len(new_text.encode('ascii'))} bytes) exceeds "
            f"HDR_SIZE ({hdr_size} bytes) for {path}. Aborting to avoid "
            f"corrupting the data payload."
        )
        raise ValueError(msg)

    if dry_run:
        print("  [dry-run] no file written")
        return

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"  backed up -> {backup_path.name}")
        else:
            print(f"  backup already exists, skipping -> {backup_path.name}")

    new_bytes = new_text.encode("ascii")
    padded = new_bytes.ljust(hdr_size, b"\x00")

    with path.open("rb+") as fp:
        fp.seek(0)
        fp.write(padded)

    print(f"  header patched ({len(new_bytes)} bytes of content, padded to {hdr_size})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing .dada files to fix",
    )
    parser.add_argument(
        "--pattern",
        default="*.dada",
        help="Glob pattern for files to match (default: *.dada)",
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
        help="Copy each file to <file>.dada.bak before editing",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"Not a directory: {args.directory}")

    glob_fn = args.directory.rglob if args.recursive else args.directory.glob
    dada_files = sorted(glob_fn(args.pattern))

    if not dada_files:
        print(f"No files matching '{args.pattern}' found in {args.directory}")
        return

    print(f"Found {len(dada_files)} file(s) to check:")
    for f in dada_files:
        print(f"  {f}")

    if args.dry_run:
        print("\n--- DRY RUN: no files will be modified ---")

    for f in dada_files:
        try:
            fix_file(f, dry_run=args.dry_run, backup=args.backup)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED on {f}: {exc}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
