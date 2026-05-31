#!/usr/bin/env python3
"""Build a CT mosaic dataset without keeping raw DICOM ZIPs locally.

This script consumes the selected MIDRC CT cohort CSV created by
download_midrc_ct_cohort.py. For each selected series it:

1. downloads one series ZIP into a temporary work folder,
2. extracts that ZIP into the temporary work folder,
3. creates one persistent mosaic PNG under data/mosaics/,
4. appends a persistent manifest row under outputs/mosaic_manifest.csv,
5. deletes the temporary ZIP and extracted DICOMs before the next series.

The durable local training artifacts are therefore the mosaic PNGs and manifest
CSV, not the raw medical image archives.

Usage:
    python scripts/build_midrc_mosaic_dataset.py
    python scripts/build_midrc_mosaic_dataset.py --limit-per-class 5
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

import create_series_mosaics as mosaics

logger = logging.getLogger("build_midrc_mosaic_dataset")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_CSV = REPO_ROOT / "data" / "metadata" / "ct_head_chest_selected.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "mosaics"
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "mosaic_manifest.csv"
DEFAULT_FAILURES = REPO_ROOT / "data" / "metadata" / "ct_mosaic_build_failures.csv"
DEFAULT_WORK_DIR = REPO_ROOT / "data" / "work" / "midrc_mosaic_build"
DEFAULT_GEN3_CLIENT = Path("/Applications/gen3-client")
DEFAULT_PROFILE = "midrc"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def clean_work_dir(work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)


def download_series_zip(
    *,
    gen3_client: Path,
    profile: str,
    object_id: str,
    raw_dir: Path,
) -> Path | None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(gen3_client),
        "download-single",
        f"--profile={profile}",
        f"--guid={object_id}",
        f"--download-path={raw_dir}",
        "--no-prompt",
        "--skip-completed",
    ]
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        logger.warning("download failed for %s", object_id)
        if result.stdout:
            logger.warning("stdout: %s", result.stdout.strip())
        if result.stderr:
            logger.warning("stderr: %s", result.stderr.strip())
        return None

    zips = sorted(raw_dir.rglob("*.zip"))
    if not zips:
        logger.warning("download finished, but no ZIP found under %s", raw_dir)
        return None
    if len(zips) > 1:
        logger.info("found %d ZIPs; using newest path", len(zips))
        zips = sorted(zips, key=lambda path: path.stat().st_mtime)
    return zips[-1]


def extract_zip(zip_path: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            info.filename = info.filename.lstrip("/").replace("..", "_")
            members.append(info)
        zf.extractall(dest_dir, members=members)
    return sum(1 for path in dest_dir.rglob("*") if path.is_file())


def one_row_series_summary(row: pd.Series, extracted_dir: Path) -> pd.DataFrame:
    series_uid = _safe_text(row.get("series_uid"))
    study_uid = _safe_text(row.get("study_uid")).strip("[]'")
    files = sorted(path for path in extracted_dir.rglob("*") if path.is_file())
    example_path = files[0] if files else ""
    return pd.DataFrame(
        [
            {
                "source_class": _safe_text(row.get("source_class")),
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "Modality": "CT",
                "BodyPartExamined": _safe_text(row.get("body_part_examined")),
                "StudyDescription": _safe_text(row.get("study_description")),
                "SeriesDescription": _safe_text(row.get("series_description")),
                "ViewPosition": "",
                "number_of_files": len(files),
                "example_path": str(example_path),
            }
        ]
    )


def load_existing_manifest(manifest_path: Path) -> pd.DataFrame:
    if manifest_path.exists():
        return pd.read_csv(manifest_path)
    return pd.DataFrame()


def write_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    if not manifest.empty and "SeriesInstanceUID" in manifest.columns:
        manifest = manifest.drop_duplicates(subset=["SeriesInstanceUID"], keep="last")
        manifest = manifest.sort_values(["body_region", "SeriesInstanceUID"])
    manifest.to_csv(manifest_path, index=False)
    return manifest


def selected_subset(
    selected: pd.DataFrame,
    *,
    limit_per_class: int | None,
) -> pd.DataFrame:
    if limit_per_class is None:
        return selected
    parts = []
    for _, group in selected.groupby("source_class", dropna=False):
        parts.append(group.head(limit_per_class))
    return pd.concat(parts, ignore_index=True) if parts else selected


def build_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(args.selected_csv)
    selected = selected_subset(selected, limit_per_class=args.limit_per_class)
    existing = pd.DataFrame() if args.reset_manifest else load_existing_manifest(args.manifest)

    output_rows: list[dict[str, Any]] = []
    if not existing.empty:
        output_rows.extend(existing.to_dict("records"))

    failures: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.failures_csv.parent.mkdir(parents=True, exist_ok=True)

    existing_by_uid = {
        _safe_text(row.get("SeriesInstanceUID")): row
        for row in output_rows
        if _safe_text(row.get("SeriesInstanceUID"))
    }

    for index, row in selected.iterrows():
        series_uid = _safe_text(row.get("series_uid"))
        source_class = _safe_text(row.get("source_class"))
        object_id = _safe_text(row.get("object_id"))
        mosaic_path = args.output_dir / f"{series_uid}.png"

        logger.info(
            "[%d/%d] %s %s",
            index + 1,
            len(selected),
            source_class,
            series_uid,
        )

        if mosaic_path.exists() and series_uid in existing_by_uid and not args.force:
            logger.info("skip existing mosaic: %s", mosaic_path)
            continue

        clean_work_dir(args.work_dir)
        raw_dir = args.work_dir / "raw" / source_class
        extracted_dir = args.work_dir / "extracted" / source_class / series_uid
        temp_summary = args.work_dir / "series_summary.csv"
        temp_manifest = args.work_dir / "mosaic_manifest.csv"

        try:
            zip_path = download_series_zip(
                gen3_client=args.gen3_client,
                profile=args.profile,
                object_id=object_id,
                raw_dir=raw_dir,
            )
            if zip_path is None:
                raise RuntimeError("download did not produce a ZIP")

            extracted_file_count = extract_zip(zip_path, extracted_dir)
            if extracted_file_count == 0:
                raise RuntimeError("ZIP extraction produced no files")

            summary = one_row_series_summary(row, extracted_dir)
            summary.to_csv(temp_summary, index=False)
            manifest = mosaics.create_mosaics(
                input_csv=temp_summary,
                dicom_root=args.work_dir / "extracted",
                output_dir=args.output_dir,
                manifest_path=temp_manifest,
                grid_size=args.grid_size,
                tile_size=args.tile_size,
                crop_content=not args.no_content_crop,
                sample_trim_fraction=args.sample_trim_fraction,
            )
            if manifest.empty:
                raise RuntimeError("mosaic generator returned no rows")

            manifest_row = manifest.iloc[0].to_dict()
            manifest_row["label_source"] = "midrc_metadata_body_region"
            manifest_row["source_object_id"] = object_id
            output_rows = [
                existing_row
                for existing_row in output_rows
                if _safe_text(existing_row.get("SeriesInstanceUID")) != series_uid
            ]
            output_rows.append(manifest_row)
            write_manifest(args.manifest, output_rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed %s: %s", series_uid, exc)
            failures.append(
                {
                    "source_class": source_class,
                    "series_uid": series_uid,
                    "object_id": object_id,
                    "error": str(exc),
                }
            )
            pd.DataFrame(failures).to_csv(args.failures_csv, index=False)
        finally:
            if not args.keep_work:
                clean_work_dir(args.work_dir)

    manifest = write_manifest(args.manifest, output_rows)
    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(args.failures_csv, index=False)
    return manifest, failures_df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--failures-csv", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--gen3-client", type=Path, default=DEFAULT_GEN3_CLIENT)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--sample-trim-fraction", type=float, default=0.10)
    parser.add_argument("--no-content-crop", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reset-manifest",
        action="store_true",
        help="Start the output manifest from this selected cohort only.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.selected_csv.exists():
        raise FileNotFoundError(f"selected cohort CSV does not exist: {args.selected_csv}")
    if not args.gen3_client.exists():
        raise FileNotFoundError(f"gen3-client does not exist: {args.gen3_client}")

    manifest, failures = build_dataset(args)

    print()
    print("=" * 72)
    print(f"  manifest rows : {len(manifest)}")
    print(f"  failures      : {len(failures)}")
    print(f"  manifest      : {args.manifest}")
    print(f"  mosaics       : {args.output_dir}")
    print(f"  work kept     : {args.keep_work}")
    print("=" * 72)
    if not manifest.empty and "body_region" in manifest.columns:
        print(manifest["body_region"].value_counts(dropna=False).to_string())

    return 0 if failures.empty else 1


if __name__ == "__main__":
    sys.exit(main())
