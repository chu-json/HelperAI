#!/usr/bin/env python3
"""Inspect a directory of DICOM files and write a per-series summary CSV.

Usage:
    python scripts/inspect_dicoms.py
    python scripts/inspect_dicoms.py --input  data/extracted
    python scripts/inspect_dicoms.py --input  data/extracted \
                                     --output outputs/sample_series_summary.csv

Design notes:
- Reads only DICOM headers (`stop_before_pixels=True`) to stay fast and avoid
  pulling pixel arrays into memory.
- Uses `force=True` so files with missing/odd preambles (common in real-world
  DICOMs) still get parsed where possible.
- Does NOT filter by `.dcm` extension because MIDRC files often have no
  extension or odd ones (e.g. just the SOPInstanceUID).
- Skips unreadable files with a logged warning rather than crashing.
- Infers `source_class` from the first directory component under the input
  root (e.g. `data/extracted/chest_xray/...` -> `chest_xray`).
- Groups files by `SeriesInstanceUID` so each row of the output represents one
  imaging series (not one file, not one patient).
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError

# Real-world MIDRC DICOMs frequently have UIDs that don't strictly conform to
# pydicom's value-representation rules. The data is still readable; the warnings
# just create noise.
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.valuerep")
logging.getLogger("pydicom").setLevel(logging.ERROR)

logger = logging.getLogger("inspect_dicoms")


DEFAULT_INPUT = Path("/Users/jason/HelperAI/data/extracted")
DEFAULT_OUTPUT = Path("/Users/jason/HelperAI/outputs/sample_series_summary.csv")


# Header fields we want to extract. PatientID is intentionally included for the
# per-file CSV (which is gitignored) but is dropped from the per-series summary.
HEADER_FIELDS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
    "BodyPartExamined",
    "StudyDescription",
    "SeriesDescription",
    "ViewPosition",
    "ImageType",
    "Rows",
    "Columns",
    "Manufacturer",
]


def _safe_get(ds: pydicom.dataset.Dataset, name: str) -> Any:
    """Return a DICOM attribute as a plain Python value, or '' if missing."""
    value = getattr(ds, name, None)
    if value is None:
        return ""
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return "\\".join(str(v) for v in value)
        except Exception:
            return str(value)
    return str(value)


def infer_source_class(path: Path, input_root: Path) -> str:
    """Infer class label from the first path component under input_root.

    e.g. /HelperAI/data/extracted/chest_xray/CASE/STUDY/SERIES/foo
         -> "chest_xray"
    Returns "" if the path is not under input_root.
    """
    try:
        rel = path.resolve().relative_to(input_root.resolve())
    except ValueError:
        return ""
    parts = rel.parts
    return parts[0] if parts else ""


def iter_candidate_files(root: Path) -> list[Path]:
    """Recursively collect candidate files. No extension filter on purpose."""
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    return [p for p in root.rglob("*") if p.is_file()]


def read_header(path: Path, input_root: Path) -> dict[str, Any] | None:
    """Read just the header of a DICOM file. Returns None on failure."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except (InvalidDicomError, OSError, Exception) as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", path, exc)
        return None

    # force=True will happily return a "dataset" for non-DICOM bytes (it just
    # has no DICOM elements). Reject those so we don't pollute the summary.
    if not getattr(ds, "SOPInstanceUID", None) and not getattr(
        ds, "SeriesInstanceUID", None
    ):
        logger.warning("Not a DICOM (no SOP/Series UID): %s", path)
        return None

    row: dict[str, Any] = {
        "path": str(path),
        "source_class": infer_source_class(path, input_root),
    }
    for field in HEADER_FIELDS:
        row[field] = _safe_get(ds, field)
    return row


def build_per_file_df(paths: list[Path], input_root: Path) -> tuple[pd.DataFrame, int]:
    """Returns (per-file df of readable DICOMs, count of unreadable files)."""
    rows = []
    n_unreadable = 0
    for p in paths:
        row = read_header(p, input_root)
        if row is None:
            n_unreadable += 1
        else:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["path", "source_class", *HEADER_FIELDS]), n_unreadable
    return pd.DataFrame(rows), n_unreadable


SERIES_COLUMNS = [
    "source_class",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality",
    "BodyPartExamined",
    "StudyDescription",
    "SeriesDescription",
    "ViewPosition",
    "number_of_files",
    "example_path",
]


def build_series_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group per-file rows by SeriesInstanceUID."""
    if df.empty or "SeriesInstanceUID" not in df.columns:
        return pd.DataFrame(columns=SERIES_COLUMNS)

    def _first_nonempty(s: pd.Series) -> Any:
        for v in s:
            if v not in ("", None):
                return v
        return ""

    grouped = df.groupby("SeriesInstanceUID", dropna=False).agg(
        source_class=("source_class", _first_nonempty),
        StudyInstanceUID=("StudyInstanceUID", _first_nonempty),
        Modality=("Modality", _first_nonempty),
        BodyPartExamined=("BodyPartExamined", _first_nonempty),
        StudyDescription=("StudyDescription", _first_nonempty),
        SeriesDescription=("SeriesDescription", _first_nonempty),
        ViewPosition=("ViewPosition", _first_nonempty),
        number_of_files=("path", "count"),
        example_path=("path", "first"),
    )
    grouped = grouped.reset_index()
    # Reorder columns to match the documented spec.
    return grouped[SERIES_COLUMNS]


def _print_summary(
    per_file_df: pd.DataFrame,
    series_df: pd.DataFrame,
    n_unreadable: int,
) -> None:
    print()
    print("=" * 60)
    print(f"  readable DICOM files : {len(per_file_df)}")
    print(f"  unreadable files     : {n_unreadable}")
    if not per_file_df.empty:
        n_studies = per_file_df["StudyInstanceUID"].replace("", pd.NA).dropna().nunique()
        n_series = per_file_df["SeriesInstanceUID"].replace("", pd.NA).dropna().nunique()
        print(f"  unique studies       : {n_studies}")
        print(f"  unique series        : {n_series}")
        print()
        print("  counts by source_class:")
        for k, v in per_file_df["source_class"].value_counts(dropna=False).items():
            print(f"    {k!s:20s} {v}")
        print()
        print("  counts by Modality:")
        for k, v in per_file_df["Modality"].value_counts(dropna=False).items():
            print(f"    {k!s:20s} {v}")
        print()
        print("  counts by StudyDescription:")
        for k, v in per_file_df["StudyDescription"].value_counts(dropna=False).items():
            print(f"    {k!s:40s} {v}")
    print("=" * 60)

    if not series_df.empty:
        print("\n  first rows of series summary:")
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 200,
            "display.max_colwidth", 60,
        ):
            print(series_df.head(10).to_string(index=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory to scan recursively (default: {DEFAULT_INPUT}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Per-series summary CSV (default: {DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "--per-file-output",
        type=Path,
        default=None,
        help=(
            "Optional per-file metadata CSV (contains PatientID; gitignored). "
            "Defaults to <output dir>/dicom_file_metadata.csv."
        ),
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging (DEBUG level).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_dir: Path = args.input
    output_csv: Path = args.output
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    per_file_csv: Path = (
        args.per_file_output
        if args.per_file_output is not None
        else output_csv.parent / "dicom_file_metadata.csv"
    )
    per_file_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning %s ...", input_dir)
    files = iter_candidate_files(input_dir)
    logger.info("Found %d candidate files.", len(files))

    per_file_df, n_unreadable = build_per_file_df(files, input_dir)
    per_file_df.to_csv(per_file_csv, index=False)
    logger.info("Wrote per-file metadata -> %s", per_file_csv)

    series_df = build_series_summary(per_file_df)
    series_df.to_csv(output_csv, index=False)
    logger.info("Wrote per-series summary (%d series) -> %s", len(series_df), output_csv)

    _print_summary(per_file_df, series_df, n_unreadable)
    return 0


if __name__ == "__main__":
    sys.exit(main())
