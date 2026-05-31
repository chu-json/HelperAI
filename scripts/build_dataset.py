#!/usr/bin/env python3
"""Filter extracted DICOMs, assign weak ground-truth labels, and build a dataset manifest.

Pipeline position:
    query_midrc_bulk.py → gen3-client → extract_midrc_zips.py → [THIS SCRIPT] → export_images.py

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --input data/extracted --output outputs/dataset_manifest.csv
    python scripts/build_dataset.py --verbose

Output:
    outputs/dataset_manifest.csv   — one row per SERIES with columns:
        series_uid, study_uid, patient_id, source_class, modality,
        body_part_label, view_label, n_files, representative_dcm,
        split (train / val / test)

Label strategy (weak ground truth from DICOM tags):
    body_part_label  — derived from BodyPartExamined → StudyDescription → Modality heuristic
    view_label       — derived from ViewPosition → SeriesDescription (X-ray only; "" for CT/MR)

Filtering:
    - Exclude series where ImageType starts with "DERIVED" (segmentation, subtraction, etc.)
    - Exclude series where SeriesDescription matches a derived-series pattern (e.g. BrainSegmentation)
    - Exclude series with no readable DICOM files
    - Exclude series where body_part_label resolves to UNKNOWN and source_class gives no hint

Patient-level split (no patient appears in more than one split):
    Train 70% / Val 15% / Test 15%
    Stratified by body_part_label so class balance is preserved across splits.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom.valuerep")
logging.getLogger("pydicom").setLevel(logging.ERROR)

logger = logging.getLogger("build_dataset")

DEFAULT_INPUT = Path("/Users/jason/HelperAI/data/extracted")
DEFAULT_OUTPUT = Path("/Users/jason/HelperAI/outputs/dataset_manifest.csv")

# ---------------------------------------------------------------------------
# Body-part label normalization
# ---------------------------------------------------------------------------
# Maps messy DICOM BodyPartExamined / StudyDescription fragments → canonical label.
# Values are upper-cased before lookup.

BODY_PART_MAP: dict[str, str] = {
    # Chest
    "CHEST": "CHEST",
    "PORT CHEST": "CHEST",
    "PORTABLE CHEST": "CHEST",
    "THORAX": "CHEST",
    "CHST": "CHEST",
    "LUNG": "CHEST",
    "LUNGS": "CHEST",
    "RIBS": "CHEST",
    "RIB": "CHEST",
    "CLAVICLE": "CHEST",
    # Abdomen
    "ABDOMEN": "ABDOMEN",
    "ABD": "ABDOMEN",
    "ABDO": "ABDOMEN",
    "ABDOMINAL": "ABDOMEN",
    "ABDO_PELVIS": "ABDOMEN",
    "ABDOPELV": "ABDOMEN",
    "ABDPELV": "ABDOMEN",
    "LIVER": "ABDOMEN",
    "GALLBLADDER": "ABDOMEN",
    "KIDNEY": "ABDOMEN",
    "KIDNEYS": "ABDOMEN",
    "SPLEEN": "ABDOMEN",
    # Pelvis (separate from abdomen for later use, mapped same for v1)
    "PELVIS": "ABDOMEN",
    "PELV": "ABDOMEN",
    "HIP": "ABDOMEN",
    "HIPS": "ABDOMEN",
    # Head / Brain
    "HEAD": "HEAD",
    "BRAIN": "HEAD",
    "SKULL": "HEAD",
    "CRANIUM": "HEAD",
    "FACE": "HEAD",
    "FACIAL": "HEAD",
    "SINUS": "HEAD",
    "SINUSES": "HEAD",
    "ORBIT": "HEAD",
    "ORBITS": "HEAD",
    "NECK": "HEAD",     # often grouped with head studies in practice
    # Spine
    "CSPINE": "SPINE",
    "CERVSPINE": "SPINE",
    "CERVICAL": "SPINE",
    "TSPINE": "SPINE",
    "THORACIC": "SPINE",
    "LSPINE": "SPINE",
    "LUMBAR": "SPINE",
    "SACRUM": "SPINE",
    "SPINE": "SPINE",
    "SPINAL": "SPINE",
    # Extremities
    "HAND": "EXTREMITY",
    "HANDS": "EXTREMITY",
    "WRIST": "EXTREMITY",
    "FOREARM": "EXTREMITY",
    "ELBOW": "EXTREMITY",
    "HUMERUS": "EXTREMITY",
    "SHOULDER": "EXTREMITY",
    "CLAVICLES": "EXTREMITY",
    "FOOT": "EXTREMITY",
    "FEET": "EXTREMITY",
    "ANKLE": "EXTREMITY",
    "LEG": "EXTREMITY",
    "TIBIA": "EXTREMITY",
    "FIBULA": "EXTREMITY",
    "KNEE": "EXTREMITY",
    "FEMUR": "EXTREMITY",
    "THIGH": "EXTREMITY",
    "FINGER": "EXTREMITY",
    "THUMB": "EXTREMITY",
    "TOE": "EXTREMITY",
}

# Keyword fragments in StudyDescription that indicate the body part,
# used as fallback when BodyPartExamined is empty.
STUDY_DESC_BODY_PART_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bchest\b|\bcxr\b|\bchr\b", re.IGNORECASE), "CHEST"),
    (re.compile(r"\bport(able)?\s+chest\b", re.IGNORECASE), "CHEST"),
    (re.compile(r"\babdomen\b|\babd\b|\babdominal\b", re.IGNORECASE), "ABDOMEN"),
    (re.compile(r"\bbrain\b|\bhead\b|\bcranial\b|\bcranium\b", re.IGNORECASE), "HEAD"),
    (re.compile(r"\bsinus\b|\bsinuses\b|\borbits?\b|\bfacial\b", re.IGNORECASE), "HEAD"),
    (re.compile(r"\bneck\b|\bcervical\b|\bc-?spine\b", re.IGNORECASE), "HEAD"),
    (re.compile(r"\bspine\b|\blumbar\b|\bthoracic\b|\bl-?spine\b|\bt-?spine\b", re.IGNORECASE), "SPINE"),
    (re.compile(r"\bknee\b|\bkneecap\b|\bpatella\b", re.IGNORECASE), "EXTREMITY"),
    (re.compile(r"\bshoulder\b|\belbow\b|\bwrist\b|\bhand\b|\bfinger\b", re.IGNORECASE), "EXTREMITY"),
    (re.compile(r"\bankle\b|\bfoot\b|\btoe\b|\bfemur\b|\btibia\b|\bfibula\b", re.IGNORECASE), "EXTREMITY"),
    (re.compile(r"\bhip\b|\bpelv", re.IGNORECASE), "ABDOMEN"),
]

# Canonical label inferred from source_class folder name when DICOM tags give nothing.
SOURCE_CLASS_TO_BODY_PART: dict[str, str] = {
    "chest_xray": "CHEST",
    "abdomen_xray": "ABDOMEN",
    "head_ct": "HEAD",
}

# ---------------------------------------------------------------------------
# View-label normalization (X-ray only)
# ---------------------------------------------------------------------------
VIEW_MAP: dict[str, str] = {
    "AP": "AP",
    "PA": "PA",
    "LL": "LATERAL",
    "RL": "LATERAL",
    "LATERAL": "LATERAL",
    "LAT": "LATERAL",
    "LT": "LATERAL",
    "RT": "LATERAL",
    "OBLIQUE": "OBLIQUE",
    "OBL": "OBLIQUE",
}

VIEW_DESC_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bpa\b|\bpa\s+and\b|\bpa/lat\b", re.IGNORECASE), "PA"),
    (re.compile(r"\bap\b|\bap\s+and\b|\bap/lat\b|\bfrontal\b", re.IGNORECASE), "AP"),
    (re.compile(r"\blateral\b|\blat\b", re.IGNORECASE), "LATERAL"),
    (re.compile(r"\boblique\b", re.IGNORECASE), "OBLIQUE"),
]

# ---------------------------------------------------------------------------
# Derived-series filter patterns
# ---------------------------------------------------------------------------
# SeriesDescription patterns that flag a series as derived / not raw imaging.
DERIVED_SERIES_DESC_PATTERNS: list[re.Pattern] = [
    re.compile(r"segmentation", re.IGNORECASE),
    re.compile(r"\bseg\b", re.IGNORECASE),
    re.compile(r"\bsr\b", re.IGNORECASE),    # Structured Report
    re.compile(r"\bkos\b", re.IGNORECASE),   # Key Object Selection
    re.compile(r"\brtdose\b|\brt dose\b", re.IGNORECASE),
    re.compile(r"\brtstruct\b|\brt struct\b", re.IGNORECASE),
    re.compile(r"\brtplan\b|\brt plan\b", re.IGNORECASE),
    re.compile(r"dose report", re.IGNORECASE),
    re.compile(r"patient protocol", re.IGNORECASE),
    re.compile(r"scout", re.IGNORECASE),     # CT scout/localiser — not diagnostic slice
    re.compile(r"localizer", re.IGNORECASE),
    re.compile(r"localiser", re.IGNORECASE),
    re.compile(r"topogram", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# DICOM reading
# ---------------------------------------------------------------------------
HEADER_FIELDS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality",
    "BodyPartExamined",
    "StudyDescription",
    "SeriesDescription",
    "ViewPosition",
    "ImageType",
    "Rows",
    "Columns",
    "InstanceNumber",
]


def _safe_str(ds: pydicom.dataset.Dataset, name: str) -> str:
    val = getattr(ds, name, None)
    if val is None:
        return ""
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        try:
            return "\\".join(str(v) for v in val)
        except Exception:
            return str(val)
    return str(val).strip()


def read_header(path: Path) -> dict[str, Any] | None:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except (InvalidDicomError, OSError, Exception):  # noqa: BLE001
        return None
    if not getattr(ds, "SOPInstanceUID", None) and not getattr(ds, "SeriesInstanceUID", None):
        return None
    return {f: _safe_str(ds, f) for f in HEADER_FIELDS} | {"path": str(path)}


def infer_source_class(path: Path, root: Path) -> str:
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
        return parts[0] if parts else ""
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# Label logic
# ---------------------------------------------------------------------------

def normalize_body_part(body_part_examined: str, study_desc: str, source_class: str) -> str:
    """Return a canonical body part label."""
    # 1. BodyPartExamined tag (clean it first)
    bpe = body_part_examined.upper().strip()
    if bpe in BODY_PART_MAP:
        return BODY_PART_MAP[bpe]

    # 2. Fragment search in BodyPartExamined (handles "CHEST-PORTABLE" etc.)
    for key, label in BODY_PART_MAP.items():
        if key in bpe:
            return label

    # 3. StudyDescription keyword search
    for pattern, label in STUDY_DESC_BODY_PART_HINTS:
        if pattern.search(study_desc):
            return label

    # 4. Source class folder name as last resort
    if source_class in SOURCE_CLASS_TO_BODY_PART:
        return SOURCE_CLASS_TO_BODY_PART[source_class]

    return "UNKNOWN"


def normalize_view(view_position: str, series_desc: str, modality: str) -> str:
    """Return AP / PA / LATERAL / OBLIQUE / '' (empty for CT/MR)."""
    if modality.upper() not in ("DX", "CR", "DR"):
        return ""

    # ViewPosition tag (most reliable)
    vp = view_position.upper().strip()
    if vp in VIEW_MAP:
        return VIEW_MAP[vp]

    # Keyword in SeriesDescription
    for pattern, label in VIEW_DESC_HINTS:
        if pattern.search(series_desc):
            return label

    return ""


def is_derived_series(series_desc: str, image_type: str, modality: str) -> bool:
    """Return True if the series should be excluded as non-original data.

    The ImageType=DERIVED check is only applied to cross-sectional modalities
    (CT, MR, PET).  For projection radiographs (DX, CR) "DERIVED" in ImageType
    simply means the image has been post-processed for presentation — that is
    normal and desirable; we should keep those images.
    """
    cross_sectional = modality.upper() in ("CT", "MR", "MRI", "PT", "NM")
    if cross_sectional and image_type:
        first_value = image_type.split("\\")[0].strip().upper()
        if first_value == "DERIVED":
            return True

    # SeriesDescription pattern matching applies to all modalities.
    for pat in DERIVED_SERIES_DESC_PATTERNS:
        if pat.search(series_desc):
            return True

    return False


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_series_df(root: Path) -> pd.DataFrame:
    """Scan root recursively; return per-series summary DataFrame."""
    all_files = [p for p in root.rglob("*") if p.is_file()]
    logger.info("Found %d candidate files under %s", len(all_files), root)

    rows: list[dict] = []
    n_bad = 0
    for p in all_files:
        rec = read_header(p)
        if rec is None:
            n_bad += 1
        else:
            rec["source_class"] = infer_source_class(p, root)
            rows.append(rec)

    logger.info("Readable: %d  Unreadable: %d", len(rows), n_bad)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Group by series: pick the representative (first) file per series.
    def _first_nonempty(s: pd.Series) -> Any:
        for v in s:
            if v not in ("", None):
                return v
        return ""

    grouped = df.groupby("SeriesInstanceUID", dropna=False).agg(
        patient_id=("PatientID", _first_nonempty),
        study_uid=("StudyInstanceUID", _first_nonempty),
        modality=("Modality", _first_nonempty),
        body_part_examined=("BodyPartExamined", _first_nonempty),
        study_description=("StudyDescription", _first_nonempty),
        series_description=("SeriesDescription", _first_nonempty),
        view_position=("ViewPosition", _first_nonempty),
        image_type=("ImageType", _first_nonempty),
        source_class=("source_class", _first_nonempty),
        n_files=("path", "count"),
        representative_dcm=("path", "first"),
    ).reset_index().rename(columns={"SeriesInstanceUID": "series_uid"})

    return grouped


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove derived / non-imaging series."""
    before = len(df)

    mask_derived = df.apply(
        lambda r: is_derived_series(r["series_description"], r["image_type"], r["modality"]),
        axis=1,
    )
    df = df[~mask_derived].copy()
    logger.info("Filtered derived series: %d removed, %d remaining", mask_derived.sum(), len(df))

    # Drop series with no files (shouldn't happen, but be safe).
    df = df[df["n_files"] > 0]

    logger.info("Total removed by all filters: %d → %d series remain", before - len(df), len(df))
    return df


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body_part_label"] = df.apply(
        lambda r: normalize_body_part(r["body_part_examined"], r["study_description"], r["source_class"]),
        axis=1,
    )
    df["view_label"] = df.apply(
        lambda r: normalize_view(r["view_position"], r["series_description"], r["modality"]),
        axis=1,
    )

    n_unknown = (df["body_part_label"] == "UNKNOWN").sum()
    if n_unknown:
        logger.warning("%d series have body_part_label=UNKNOWN (consider widening BODY_PART_MAP)", n_unknown)

    return df


# ---------------------------------------------------------------------------
# Patient-level stratified split
# ---------------------------------------------------------------------------

def assign_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15) -> pd.DataFrame:
    """Assign train/val/test at the patient level, stratified by body_part_label.

    A patient never appears in more than one split, preventing data leakage
    between slices / views of the same patient.
    """
    df = df.copy()
    df["split"] = "train"

    # Get unique (patient_id, body_part_label) pairs.
    patient_label = (
        df[["patient_id", "body_part_label"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # For each label, shuffle patients deterministically (seeded) and split.
    for label, group in patient_label.groupby("body_part_label"):
        pids = group["patient_id"].tolist()
        # Deterministic shuffle based on a hash of the patient_id string.
        pids.sort(key=lambda p: hashlib.md5(str(p).encode()).hexdigest())

        n = len(pids)
        n_val = max(1, round(n * val_frac))
        n_test = max(1, round(n * test_frac))

        val_pids = set(pids[:n_val])
        test_pids = set(pids[n_val: n_val + n_test])

        mask_val = df["patient_id"].isin(val_pids) & (df["body_part_label"] == label)
        mask_test = df["patient_id"].isin(test_pids) & (df["body_part_label"] == label)
        df.loc[mask_val, "split"] = "val"
        df.loc[mask_test, "split"] = "test"

    split_counts = df.groupby(["body_part_label", "split"]).size().unstack(fill_value=0)
    logger.info("Split distribution:\n%s", split_counts.to_string())
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Extracted DICOM root (default: {DEFAULT_INPUT}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Dataset manifest CSV (default: {DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "--drop-unknown",
        action="store_true",
        default=False,
        help="Drop series whose body_part_label cannot be resolved (UNKNOWN). Default: keep them.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of patients to put in val (default: 0.15).",
    )
    p.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Fraction of patients to put in test (default: 0.15).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    root: Path = args.input.resolve()
    out_csv: Path = args.output
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning %s …", root)
    series_df = build_series_df(root)
    if series_df.empty:
        logger.error("No readable DICOM series found under %s.", root)
        return 1

    series_df = apply_filters(series_df)
    series_df = assign_labels(series_df)

    if args.drop_unknown:
        before = len(series_df)
        series_df = series_df[series_df["body_part_label"] != "UNKNOWN"]
        logger.info("Dropped %d UNKNOWN-label series.", before - len(series_df))

    series_df = assign_split(series_df, val_frac=args.val_frac, test_frac=args.test_frac)

    # Final column order for the manifest.
    cols = [
        "series_uid", "study_uid", "patient_id",
        "source_class", "modality",
        "body_part_label", "view_label",
        "body_part_examined", "study_description", "series_description",
        "n_files", "representative_dcm", "split",
    ]
    series_df = series_df[[c for c in cols if c in series_df.columns]]
    series_df.to_csv(out_csv, index=False)
    logger.info("Wrote dataset manifest (%d series) → %s", len(series_df), out_csv)

    # Print summary.
    print()
    print("=" * 60)
    print(f"  Total series          : {len(series_df)}")
    print(f"  Unique patients       : {series_df['patient_id'].nunique()}")
    print()
    print("  By body_part_label:")
    for k, v in series_df["body_part_label"].value_counts().items():
        print(f"    {k:20s} {v:5d}")
    print()
    print("  By split:")
    for k, v in series_df["split"].value_counts().items():
        print(f"    {k:10s} {v:5d}")
    print()
    print("  For X-ray view_label:")
    xray = series_df[series_df["modality"].isin(["DX", "CR", "DR"])]
    for k, v in xray["view_label"].value_counts(dropna=False).items():
        print(f"    {str(k):15s} {v:5d}")
    print()
    print(f"  Manifest → {out_csv}")
    print()
    print("  Next step:")
    print("    python scripts/export_images.py")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
