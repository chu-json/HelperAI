#!/usr/bin/env python3
"""Convert DICOM series to PNG images ready for ResNet training.

Pipeline position:
    build_dataset.py → [THIS SCRIPT] → ResNet training

Usage:
    python scripts/export_images.py
    python scripts/export_images.py --manifest outputs/dataset_manifest.csv
    python scripts/export_images.py --size 224 --ct-window brain
    python scripts/export_images.py --workers 8

Output layout (PyTorch ImageFolder-compatible):
    data/images/
        train/
            CHEST/   HEAD/   ABDOMEN/   SPINE/   EXTREMITY/
        val/
            ...
        test/
            ...

Also writes:
    outputs/image_manifest.csv  — same as dataset_manifest.csv but with
                                   an added `image_path` column pointing to
                                   the exported PNG.

Per-modality processing:
    X-ray (DX / CR / DR):
        - Single DICOM file per series.
        - Handles PhotometricInterpretation MONOCHROME1 (inverted → flip).
        - Rescale pixel values to [0, 255] using 1st / 99th percentile clip.
        - Resize to --size × --size (default 224).

    CT:
        - Multi-slice series: sort by InstanceNumber, pick the middle third,
          choose the slice with the highest variance (most informative anatomy).
        - Apply windowing (Window Center / Width from DICOM tags or defaults).
        - Supports preset windows via --ct-window: brain | soft_tissue | lung | bone
        - Normalize windowed output to [0, 255].
        - Resize to --size × --size.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.valuerep")
logging.getLogger("pydicom").setLevel(logging.ERROR)

try:
    from PIL import Image
except ImportError:
    print("Pillow is required: pip install Pillow")
    sys.exit(1)

logger = logging.getLogger("export_images")

DEFAULT_MANIFEST = Path("/Users/jason/HelperAI/outputs/dataset_manifest.csv")
DEFAULT_IMAGE_DIR = Path("/Users/jason/HelperAI/data/images")
DEFAULT_IMAGE_MANIFEST = Path("/Users/jason/HelperAI/outputs/image_manifest.csv")

DEFAULT_SIZE = 224

# ---------------------------------------------------------------------------
# CT window presets  (center, width)
# ---------------------------------------------------------------------------
CT_WINDOWS: dict[str, tuple[int, int]] = {
    "brain":       (40,   80),
    "subdural":    (75,  215),
    "stroke":      (32,   8),
    "soft_tissue": (50,  400),
    "bone":        (400, 1800),
    "lung":        (-600, 1500),
    "chest":       (40,  400),
}

# Modalities treated as projection radiographs.
XRAY_MODALITIES = {"DX", "CR", "DR", "RG", "RF"}

# Modalities treated as cross-sectional (slice-based).
CT_MODALITIES = {"CT"}
MR_MODALITIES = {"MR", "MRI"}


# ---------------------------------------------------------------------------
# DICOM pixel helpers
# ---------------------------------------------------------------------------

def _load_pixel_array(path: Path) -> tuple[np.ndarray | None, pydicom.dataset.Dataset | None]:
    """Load pixel array from a DICOM file. Returns (array, dataset) or (None, None)."""
    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
        return arr, ds
    except (InvalidDicomError, AttributeError, Exception) as exc:  # noqa: BLE001
        logger.debug("Could not load pixel array from %s: %s", path, exc)
        return None, None


def _apply_rescale(arr: np.ndarray, ds: pydicom.dataset.Dataset) -> np.ndarray:
    """Apply RescaleSlope / RescaleIntercept to convert to Hounsfield (CT) or stored units."""
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


def _apply_ct_window(arr: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply CT window (center/width) and normalize to [0, 255]."""
    lo = center - width / 2
    hi = center + width / 2
    windowed = np.clip(arr, lo, hi)
    # Normalize to [0, 255]
    windowed = (windowed - lo) / (hi - lo) * 255.0
    return windowed.astype(np.uint8)


def _normalize_xray(arr: np.ndarray, ds: pydicom.dataset.Dataset) -> np.ndarray:
    """Normalize X-ray pixel array to [0, 255] uint8."""
    # Handle MONOCHROME1: high pixel value = dark (inverted vs convention)
    pi = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if str(pi).upper().strip() == "MONOCHROME1":
        arr = arr.max() - arr

    # Clip at 1st / 99th percentile to suppress outlier pixels.
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1  # avoid div-by-zero for blank images
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def _to_pil(arr: np.ndarray, target_size: int) -> Image.Image:
    """Convert a 2-D uint8 array to a square RGB PIL image."""
    if arr.ndim == 3:
        # Some DICOMs (rare) return (H, W, C) — take the first channel.
        arr = arr[:, :, 0]
    img = Image.fromarray(arr, mode="L").convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Series processors
# ---------------------------------------------------------------------------

def _dcm_files_sorted(series_dir: str | Path) -> list[Path]:
    """Return all DICOM files in a series directory, sorted by InstanceNumber."""
    base = Path(series_dir) if isinstance(series_dir, str) else series_dir
    if base.is_file():
        # representative_dcm might point to a single file; use its parent.
        base = base.parent

    files = [p for p in base.rglob("*") if p.is_file()]
    if not files:
        return []

    def _sort_key(p: Path) -> int:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            return int(getattr(ds, "InstanceNumber", 0) or 0)
        except Exception:  # noqa: BLE001
            return 0

    return sorted(files, key=_sort_key)


def process_xray(
    representative_dcm: str,
    target_size: int,
) -> np.ndarray | None:
    """Process a single X-ray DICOM file → uint8 H×W array."""
    path = Path(representative_dcm)
    arr, ds = _load_pixel_array(path)
    if arr is None:
        return None
    if arr.ndim > 2:
        arr = arr[0] if arr.ndim == 3 else arr  # take first frame if multi-frame
    return _normalize_xray(arr, ds)


def process_ct(
    representative_dcm: str,
    target_size: int,
    window_center: int,
    window_width: int,
) -> np.ndarray | None:
    """Process a CT series → uint8 H×W array (best representative slice)."""
    path = Path(representative_dcm)
    series_files = _dcm_files_sorted(path)
    if not series_files:
        return None

    # Focus on the middle third of the stack to avoid top/bottom non-diagnostic slices.
    n = len(series_files)
    start = n // 3
    end = 2 * n // 3 + 1
    candidates = series_files[start:end] or series_files  # fallback: full stack

    # Pick the slice with the highest variance within the candidates.
    best_arr: np.ndarray | None = None
    best_ds: Any = None
    best_var = -1.0

    for p in candidates:
        arr, ds = _load_pixel_array(p)
        if arr is None or arr.ndim != 2:
            continue
        v = float(np.var(arr))
        if v > best_var:
            best_var = v
            best_arr = arr
            best_ds = ds

    if best_arr is None:
        return None

    # Apply DICOM rescale before windowing.
    best_arr = _apply_rescale(best_arr, best_ds)

    # Use DICOM-embedded window if available and we're using the defaults.
    wc = float(getattr(best_ds, "WindowCenter", None) or window_center)
    ww = float(getattr(best_ds, "WindowWidth", None) or window_width)
    # WindowCenter/Width can be a list; take the first element.
    if hasattr(wc, "__iter__"):
        wc = float(list(wc)[0])
    if hasattr(ww, "__iter__"):
        ww = float(list(ww)[0])

    return _apply_ct_window(best_arr, wc, ww)


def process_mr(
    representative_dcm: str,
    target_size: int,
) -> np.ndarray | None:
    """Process an MR series — same as X-ray (percentile normalization, best slice)."""
    path = Path(representative_dcm)
    series_files = _dcm_files_sorted(path)
    if not series_files:
        return None

    n = len(series_files)
    start, end = n // 3, 2 * n // 3 + 1
    candidates = series_files[start:end] or series_files

    best_arr: np.ndarray | None = None
    best_ds: Any = None
    best_var = -1.0
    for p in candidates:
        arr, ds = _load_pixel_array(p)
        if arr is None or arr.ndim != 2:
            continue
        v = float(np.var(arr))
        if v > best_var:
            best_var = v
            best_arr = arr
            best_ds = ds

    if best_arr is None:
        return None
    return _normalize_xray(best_arr, best_ds)


# ---------------------------------------------------------------------------
# Per-row export
# ---------------------------------------------------------------------------

def export_series(
    row: pd.Series,
    image_dir: Path,
    target_size: int,
    ct_window_center: int,
    ct_window_width: int,
) -> str | None:
    """Convert one series row → PNG. Returns output path str or None on failure."""
    label = row["body_part_label"]
    split = row["split"]
    series_uid = str(row["series_uid"])
    modality = str(row.get("modality", "")).upper()
    rep_dcm = str(row["representative_dcm"])

    out_dir = image_dir / split / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize series_uid for use as filename.
    safe_uid = series_uid.replace(".", "_").replace("/", "_")[:80]
    out_path = out_dir / f"{safe_uid}.png"

    if out_path.exists():
        return str(out_path)

    if modality in XRAY_MODALITIES:
        arr = process_xray(rep_dcm, target_size)
    elif modality in CT_MODALITIES:
        arr = process_ct(rep_dcm, target_size, ct_window_center, ct_window_width)
    elif modality in MR_MODALITIES:
        arr = process_mr(rep_dcm, target_size)
    else:
        # Unknown modality: try X-ray path as it's the most generic.
        arr = process_xray(rep_dcm, target_size)

    if arr is None:
        logger.warning("Could not produce image for series %s (%s)", series_uid, rep_dcm)
        return None

    try:
        img = _to_pil(arr, target_size)
        img.save(str(out_path), format="PNG", optimize=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save PNG for %s: %s", series_uid, exc)
        return None

    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Dataset manifest CSV from build_dataset.py (default: {DEFAULT_MANIFEST}).",
    )
    p.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Root output dir for PNG images (default: {DEFAULT_IMAGE_DIR}).",
    )
    p.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_IMAGE_MANIFEST,
        help=f"Image manifest CSV to write (default: {DEFAULT_IMAGE_MANIFEST}).",
    )
    p.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Output image size in pixels (square, default: {DEFAULT_SIZE}).",
    )
    p.add_argument(
        "--ct-window",
        choices=list(CT_WINDOWS.keys()),
        default="soft_tissue",
        help="CT window preset to use when DICOM has no embedded window (default: soft_tissue).",
    )
    p.add_argument(
        "--ct-window-center",
        type=int,
        default=None,
        help="Override CT window center (overrides --ct-window).",
    )
    p.add_argument(
        "--ct-window-width",
        type=int,
        default=None,
        help="Override CT window width (overrides --ct-window).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads (default: 4).",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        metavar="SPLIT",
        help="Which splits to export (default: train val test).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest_path: Path = args.manifest
    if not manifest_path.exists():
        logger.error("Manifest not found: %s\nRun build_dataset.py first.", manifest_path)
        return 1

    df = pd.read_csv(manifest_path)
    logger.info("Loaded manifest: %d series", len(df))

    # Filter to requested splits.
    df = df[df["split"].isin(args.splits)].copy()
    logger.info("After split filter: %d series", len(df))

    if df.empty:
        logger.error("No series match the requested splits: %s", args.splits)
        return 1

    # Resolve CT window.
    wc_preset, ww_preset = CT_WINDOWS[args.ct_window]
    ct_wc = args.ct_window_center if args.ct_window_center is not None else wc_preset
    ct_ww = args.ct_window_width if args.ct_window_width is not None else ww_preset
    logger.info("CT window: center=%d  width=%d  (preset: %s)", ct_wc, ct_ww, args.ct_window)

    image_dir: Path = args.image_dir
    image_dir.mkdir(parents=True, exist_ok=True)

    # Export with thread pool.
    image_paths: list[str | None] = [None] * len(df)
    rows_list = list(df.iterrows())

    logger.info("Exporting %d series with %d workers …", len(df), args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                export_series,
                row,
                image_dir,
                args.size,
                ct_wc,
                ct_ww,
            ): idx
            for idx, (_, row) in enumerate(rows_list)
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                image_paths[idx] = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Worker error for row %d: %s", idx, exc)
            done += 1
            if done % 100 == 0 or done == len(futures):
                logger.info("Progress: %d / %d", done, len(futures))

    df["image_path"] = image_paths
    n_ok = df["image_path"].notna().sum()
    n_fail = len(df) - n_ok
    logger.info("Exported: %d OK  %d failed", n_ok, n_fail)

    # Write image manifest.
    out_manifest: Path = args.output_manifest
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)
    logger.info("Wrote image manifest → %s", out_manifest)

    # Print summary.
    print()
    print("=" * 60)
    print(f"  Series processed      : {len(df)}")
    print(f"  Images exported (OK)  : {n_ok}")
    print(f"  Failed                : {n_fail}")
    print(f"  Image size            : {args.size}×{args.size} px RGB PNG")
    print(f"  Output dir            : {image_dir}")
    print(f"  Image manifest        : {out_manifest}")
    print()
    print("  Class counts (train/val/test):")
    if "body_part_label" in df.columns and "split" in df.columns:
        pivot = (
            df[df["image_path"].notna()]
            .groupby(["body_part_label", "split"])
            .size()
            .unstack(fill_value=0)
        )
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(pivot.to_string())
    print()
    print("  ImageFolder-compatible layout:")
    print(f"    {image_dir}/train/<LABEL>/<series>.png")
    print(f"    {image_dir}/val/<LABEL>/<series>.png")
    print(f"    {image_dir}/test/<LABEL>/<series>.png")
    print("=" * 60)

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
