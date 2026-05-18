#!/usr/bin/env python3
"""Extract MIDRC-downloaded series ZIPs into a parallel `data/extracted/` tree.

MIDRC delivers one DICOM series per ZIP. The Gen3 download layout looks like:

    data/raw/<class>/<case_id>/<study_uid>/<series_uid>.zip

This script extracts each ZIP to:

    data/extracted/<class>/<case_id>/<study_uid>/<series_uid>/

so that:
- the class label (chest_xray / head_ct / etc.) is preserved as the top folder,
- the study/series provenance is preserved on disk,
- each series sits in its own directory (useful for series-level grouping later).

Usage:
    python scripts/extract_midrc_zips.py
    python scripts/extract_midrc_zips.py --input data/raw --output data/extracted
    python scripts/extract_midrc_zips.py --force        # re-extract even if dest exists
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger("extract_midrc_zips")


def find_zips(root: Path) -> list[Path]:
    """Recursively find all .zip files under root (case-insensitive)."""
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".zip")


def extract_dir_for(zip_path: Path, input_root: Path, output_root: Path) -> Path:
    """Mirror the zip's relative path under output_root, dropping the .zip suffix.

    e.g. data/raw/chest_xray/CASE/STUDY/SERIES.zip
         -> data/extracted/chest_xray/CASE/STUDY/SERIES/
    """
    rel = zip_path.relative_to(input_root)
    return output_root / rel.with_suffix("")


def extract_one(
    zip_path: Path,
    dest_dir: Path,
    *,
    force: bool = False,
) -> tuple[bool, int]:
    """Extract a single zip. Returns (was_extracted, num_files_inside)."""
    if dest_dir.exists() and any(dest_dir.iterdir()) and not force:
        existing = sum(1 for _ in dest_dir.rglob("*") if _.is_file())
        logger.info("skip (already extracted, %d files): %s", existing, dest_dir)
        return False, existing

    if force and dest_dir.exists():
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            # zipfile.extractall is safe against absolute paths in Python 3.6.2+
            # but we still defensively skip entries that try to escape dest_dir.
            members = []
            for info in zf.infolist():
                # Skip directory entries; we create dirs implicitly.
                if info.is_dir():
                    continue
                # Defensive: drop any leading "/" or ".." traversal.
                name = info.filename.lstrip("/").replace("..", "_")
                info.filename = name
                members.append(info)
            zf.extractall(dest_dir, members=members)
    except zipfile.BadZipFile as exc:
        logger.warning("BAD ZIP: %s (%s)", zip_path, exc)
        return False, 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("FAILED to extract %s: %s", zip_path, exc)
        return False, 0

    n = sum(1 for _ in dest_dir.rglob("*") if _.is_file())
    logger.info("extracted %d files -> %s", n, dest_dir)
    return True, n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/jason/HelperAI/data/raw"),
        help="Root to search for .zip files (default: data/raw).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/jason/HelperAI/data/extracted"),
        help="Root to extract into (default: data/extracted).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if the destination directory is non-empty.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_root: Path = args.input.resolve()
    output_root: Path = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    zips = find_zips(input_root)
    logger.info("Found %d zip file(s) under %s", len(zips), input_root)

    n_extracted = 0
    n_files_total = 0
    for zp in zips:
        dest = extract_dir_for(zp, input_root, output_root)
        was, nfiles = extract_one(zp, dest, force=args.force)
        if was:
            n_extracted += 1
        n_files_total += nfiles

    # Final tally: count everything actually on disk under output_root.
    extracted_files = [p for p in output_root.rglob("*") if p.is_file()]

    print()
    print("=" * 60)
    print(f"  zips found        : {len(zips)}")
    print(f"  zips extracted    : {n_extracted} (others were skipped or already done)")
    print(f"  files on disk     : {len(extracted_files)} (under {output_root})")
    print("=" * 60)
    print("  first 10 extracted file paths:")
    for p in extracted_files[:10]:
        print("   ", p)

    return 0


if __name__ == "__main__":
    sys.exit(main())
