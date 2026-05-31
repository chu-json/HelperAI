#!/usr/bin/env python3
"""Create fixed-size CT series mosaics from extracted DICOM slices.

This is the reusable pipeline entrypoint for the "series-level mosaic"
representation. Each CT series is summarized as one grayscale PNG made from
evenly sampled slices across the ordered volume.

Default output:
    data/mosaics/<SeriesInstanceUID>.png
    outputs/mosaic_manifest.csv

Usage:
    python scripts/create_series_mosaics.py
    python scripts/create_series_mosaics.py --tile-size 256
    python scripts/create_series_mosaics.py --no-content-crop
    python scripts/create_series_mosaics.py --sample-trim-fraction 0
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.valuerep")

logger = logging.getLogger("create_series_mosaics")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = REPO_ROOT / "outputs" / "sample_series_summary.csv"
DEFAULT_DICOM_ROOT = REPO_ROOT / "data" / "extracted"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "mosaics"
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "mosaic_manifest.csv"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def infer_body_region(row: pd.Series) -> str:
    """Infer body region conservatively from current metadata labels/text."""
    source = _safe_text(row.get("source_class")).lower()
    text = " ".join(
        _safe_text(row.get(col)).lower()
        for col in ("BodyPartExamined", "StudyDescription", "SeriesDescription")
    )
    if "head" in source or "brain" in source or "head" in text or "brain" in text:
        return "head"
    if "chest" in source or "chest" in text or "thorax" in text:
        return "chest"
    return "unknown"


def infer_contrast_usage(row: pd.Series) -> str:
    """Infer contrast usage only when the metadata text is explicit."""
    text = " ".join(
        _safe_text(row.get(col)).upper()
        for col in ("StudyDescription", "SeriesDescription")
    )
    without_markers = (
        "W/O",
        "WITHOUT",
        "NON-CONTRAST",
        "NONCONTRAST",
        "NO CONTRAST",
    )
    if any(marker in text for marker in without_markers):
        return "without_contrast"
    if "CONTRAST" in text or " W/ " in f" {text} " or text.startswith("W/"):
        return "with_contrast"
    return "unknown"


def _position_sort_value(ds: pydicom.dataset.Dataset) -> float | None:
    position = getattr(ds, "ImagePositionPatient", None)
    if position is None or len(position) < 3:
        return None
    try:
        return float(position[2])
    except (TypeError, ValueError):
        return None


def _instance_sort_value(ds: pydicom.dataset.Dataset) -> int | None:
    instance = getattr(ds, "InstanceNumber", None)
    if instance is None:
        return None
    try:
        return int(instance)
    except (TypeError, ValueError):
        return None


def sort_dicom_files(paths: list[Path]) -> list[Path]:
    """Sort slices by patient position, then instance number, then filename."""
    sort_rows = []
    for path in paths:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            position = _position_sort_value(ds)
            instance = _instance_sort_value(ds)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read header for sorting %s: %s", path, exc)
            position = None
            instance = None
        sort_rows.append(
            (
                position is None,
                position if position is not None else 0.0,
                instance is None,
                instance if instance is not None else 0,
                str(path),
                path,
            )
        )
    return [row[-1] for row in sorted(sort_rows)]


def collect_series_files(series_uid: str, dicom_root: Path) -> list[Path]:
    """Find DICOM files for a series using the extracted MIDRC path layout."""
    files = [
        path
        for path in dicom_root.rglob("*.dcm")
        if path.is_file() and series_uid in str(path)
    ]
    return sort_dicom_files(files)


def sample_evenly(
    paths: list[Path],
    count: int,
    *,
    trim_fraction: float = 0.0,
) -> list[Path]:
    """Pick count files at evenly spaced indices across the series volume.

    trim_fraction removes the same fraction from the beginning and end before
    sampling. A value of 0.10 samples the middle 80% of the ordered volume.
    """
    if not paths:
        return []
    if trim_fraction < 0 or trim_fraction >= 0.5:
        raise ValueError("trim_fraction must be >= 0 and < 0.5")

    last_index = len(paths) - 1
    start_index = int(round(last_index * trim_fraction))
    end_index = int(round(last_index * (1 - trim_fraction)))
    if start_index > end_index:
        start_index = 0
        end_index = last_index

    indices = np.linspace(start_index, end_index, count)
    return [paths[int(round(index))] for index in indices]


def window_ct_pixels(
    ds: pydicom.dataset.Dataset,
    *,
    center: float,
    width: float,
) -> np.ndarray:
    """Load, rescale, window, and normalize CT pixels to uint8 grayscale."""
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    low = center - width / 2
    high = center + width / 2
    arr = np.clip(arr, low, high)
    arr = (arr - low) / max(high - low, 1e-6)
    return (arr * 255).astype(np.uint8)


def crop_to_content(pixels: np.ndarray, *, margin_fraction: float = 0.04) -> np.ndarray:
    """Remove empty black border so each tile spends pixels on anatomy."""
    mask = pixels > 0
    if not mask.any():
        return pixels

    ys, xs = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    height, width = pixels.shape
    margin = int(round(max(height, width) * margin_fraction))
    y_min = max(0, y_min - margin)
    y_max = min(height - 1, y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(width - 1, x_max + margin)

    return pixels[y_min : y_max + 1, x_min : x_max + 1]


def ct_window_for_region(body_region: str) -> tuple[float, float]:
    """Return a simple default CT window for the inferred body region."""
    if body_region == "chest":
        return -600.0, 1500.0
    # Current data is head CT. Brain window keeps soft tissue contrast visible.
    return 40.0, 80.0


def make_mosaic(
    sampled_paths: list[Path],
    *,
    body_region: str,
    grid_size: int,
    tile_size: int,
    crop_content: bool = True,
) -> Image.Image:
    """Build a square grayscale mosaic from sampled DICOM paths."""
    center, width = ct_window_for_region(body_region)
    canvas_size = grid_size * tile_size
    mosaic = Image.new("L", (canvas_size, canvas_size), color=0)

    for index, path in enumerate(sampled_paths[: grid_size * grid_size]):
        ds = pydicom.dcmread(str(path), force=True)
        pixels = window_ct_pixels(ds, center=center, width=width)
        if crop_content:
            pixels = crop_to_content(pixels)
        tile = Image.fromarray(pixels, mode="L").resize(
            (tile_size, tile_size),
            resample=Image.Resampling.BILINEAR,
        )
        row = index // grid_size
        col = index % grid_size
        mosaic.paste(tile, (col * tile_size, row * tile_size))

    return mosaic


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def create_mosaics(
    *,
    input_csv: Path,
    dicom_root: Path,
    output_dir: Path,
    manifest_path: Path,
    grid_size: int,
    tile_size: int,
    crop_content: bool = True,
    sample_trim_fraction: float = 0.10,
) -> pd.DataFrame:
    """Generate CT mosaics and return the manifest DataFrame."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input series summary does not exist: {input_csv}")
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root does not exist: {dicom_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    series_df = pd.read_csv(input_csv)
    ct_df = series_df[series_df["Modality"].astype(str).str.upper() == "CT"].copy()

    rows: list[dict[str, Any]] = []
    sample_count = grid_size * grid_size
    mosaic_size = grid_size * tile_size

    for _, series in ct_df.iterrows():
        series_uid = _safe_text(series.get("SeriesInstanceUID"))
        if not series_uid:
            logger.warning("Skipping CT row with no SeriesInstanceUID")
            continue

        files = collect_series_files(series_uid, dicom_root)
        if not files:
            logger.warning("No DICOM files found for series %s", series_uid)
            continue

        body_region = infer_body_region(series)
        contrast_usage = infer_contrast_usage(series)
        sampled = sample_evenly(
            files,
            sample_count,
            trim_fraction=sample_trim_fraction,
        )
        mosaic = make_mosaic(
            sampled,
            body_region=body_region,
            grid_size=grid_size,
            tile_size=tile_size,
            crop_content=crop_content,
        )

        mosaic_path = output_dir / f"{series_uid}.png"
        mosaic.save(mosaic_path)

        rows.append(
            {
                "SeriesInstanceUID": series_uid,
                "StudyInstanceUID": _safe_text(series.get("StudyInstanceUID")),
                "source_class": _safe_text(series.get("source_class")),
                "body_region": body_region,
                "contrast_usage": contrast_usage,
                "modality": "CT",
                "number_of_files": len(files),
                "sampled_slice_count": len(sampled),
                "sample_trim_fraction": sample_trim_fraction,
                "grid_size": grid_size,
                "tile_size": tile_size,
                "mosaic_width": mosaic_size,
                "mosaic_height": mosaic_size,
                "mosaic_path": _relative_path(mosaic_path),
            }
        )
        logger.info("Wrote mosaic for series %s -> %s", series_uid, mosaic_path)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--dicom-root", type=Path, default=DEFAULT_DICOM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument(
        "--sample-trim-fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction to remove from both ends of the ordered CT volume before "
            "sampling. Default 0.10 samples the middle 80%."
        ),
    )
    parser.add_argument(
        "--no-content-crop",
        action="store_true",
        help="Resize each full CT frame without cropping black background first.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest = create_mosaics(
        input_csv=args.input_csv.resolve(),
        dicom_root=args.dicom_root.resolve(),
        output_dir=args.output_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        grid_size=args.grid_size,
        tile_size=args.tile_size,
        crop_content=not args.no_content_crop,
        sample_trim_fraction=args.sample_trim_fraction,
    )

    print()
    print("=" * 60)
    print(f"  CT mosaics written : {len(manifest)}")
    print(f"  output directory   : {args.output_dir}")
    print(f"  manifest           : {args.manifest}")
    print("=" * 60)
    if not manifest.empty:
        print(manifest.head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
