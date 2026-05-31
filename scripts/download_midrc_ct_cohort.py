#!/usr/bin/env python3
"""Query and optionally download a balanced MIDRC CT head/chest cohort.

This script is the V1 data acquisition entrypoint for the binary body-region
model. It ignores contrast as a training target and keeps only CT series that
can be conservatively labeled as head/brain or chest/thorax from metadata.

Default behavior is metadata-only. Add --download to fetch the selected series
ZIPs with gen3-client.

Usage:
    python scripts/download_midrc_ct_cohort.py
    python scripts/download_midrc_ct_cohort.py --target-per-class 100 --download
    python scripts/download_midrc_ct_cohort.py --use-existing-selection --download
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from gen3.auth import Gen3Auth
from gen3.query import Gen3Query

logger = logging.getLogger("download_midrc_ct_cohort")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API = "https://data.midrc.org"
DEFAULT_CRED = Path.home() / "Downloads" / "credentials.json"
DEFAULT_GEN3_CLIENT = Path("/Applications/gen3-client")
DEFAULT_PROFILE = "midrc"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_METADATA_DIR = REPO_ROOT / "data" / "metadata"

RETURN_FIELDS = [
    "object_id",
    "file_name",
    "file_size",
    "data_format",
    "data_type",
    "data_category",
    "modality",
    "study_description",
    "series_description",
    "body_part_examined",
    "study_uid",
    "series_uid",
    "case_ids",
]

HEAD_MARKERS = ("head", "brain", "skull")
CHEST_MARKERS = ("chest", "thorax", "lung", "pulmonary")
DICOM_DATA_TYPE = "DICOM"
DICOM_DATA_FORMAT = "DCM"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(v) for v in value)
    return str(value)


def infer_body_region(record: dict[str, Any]) -> str:
    """Return head/chest when metadata is clear, otherwise unknown."""
    text = " ".join(
        _safe_text(record.get(field)).lower()
        for field in (
            "body_part_examined",
            "study_description",
            "series_description",
            "file_name",
        )
    )
    has_head = any(marker in text for marker in HEAD_MARKERS)
    has_chest = any(marker in text for marker in CHEST_MARKERS)
    if has_head and not has_chest:
        return "head"
    if has_chest and not has_head:
        return "chest"
    return "unknown"


def class_name_for_region(body_region: str) -> str:
    if body_region == "head":
        return "head_ct"
    if body_region == "chest":
        return "chest_ct"
    return "unknown_ct"


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {field: record.get(field, "") for field in RETURN_FIELDS}
    body_region = infer_body_region(row)
    row["body_region"] = body_region
    row["source_class"] = class_name_for_region(body_region)
    return row


def is_dicom_record(row: pd.Series) -> bool:
    return (
        _safe_text(row.get("data_type")).upper() == DICOM_DATA_TYPE
        and _safe_text(row.get("data_format")).upper() == DICOM_DATA_FORMAT
    )


def query_ct_records(
    query: Gen3Query,
    *,
    page_size: int,
    max_records: int,
) -> list[dict[str, Any]]:
    """Query CT data_file records from MIDRC in pages."""
    filter_object = {"AND": [{"IN": {"modality": ["CT"]}}]}
    records: list[dict[str, Any]] = []
    offset = 0

    while len(records) < max_records:
        logger.info("Querying CT metadata offset=%d first=%d", offset, page_size)
        page = query.raw_data_download(
            data_type="data_file",
            fields=RETURN_FIELDS,
            filter_object=filter_object,
            first=page_size,
            offset=offset,
        )
        page = page or []
        if not page:
            break
        records.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    return records[:max_records]


def build_selected_cohort(
    records: list[dict[str, Any]],
    *,
    target_per_class: int,
    exclude_object_ids: set[str] | None = None,
    exclude_series_uids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize records and select unique head/chest CT series."""
    normalized = [normalize_record(record) for record in records]
    candidates = pd.DataFrame(normalized)
    if candidates.empty:
        return candidates, candidates

    exclude_object_ids = exclude_object_ids or set()
    exclude_series_uids = exclude_series_uids or set()

    candidates = candidates[
        candidates["body_region"].isin(["head", "chest"])
        & candidates.apply(is_dicom_record, axis=1)
    ].copy()
    if exclude_object_ids:
        candidates = candidates[
            ~candidates["object_id"].map(_safe_text).isin(exclude_object_ids)
        ]
    if exclude_series_uids:
        candidates = candidates[
            ~candidates["series_uid"].map(_safe_text).isin(exclude_series_uids)
        ]
    candidates = candidates.dropna(subset=["object_id", "series_uid"])
    candidates = candidates.drop_duplicates(subset=["series_uid"], keep="first")

    selected_parts = []
    for body_region in ("head", "chest"):
        part = candidates[candidates["body_region"] == body_region].head(target_per_class)
        selected_parts.append(part)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else candidates
    return candidates.reset_index(drop=True), selected.reset_index(drop=True)


def write_metadata_outputs(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    metadata_dir: Path,
) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(metadata_dir / "ct_head_chest_candidates.csv", index=False)
    selected.to_csv(metadata_dir / "ct_head_chest_selected.csv", index=False)

    for source_class, group in selected.groupby("source_class", dropna=False):
        group.to_csv(metadata_dir / f"{source_class}_selected.csv", index=False)

    summary = (
        selected.groupby(["source_class", "body_region"], dropna=False)
        .size()
        .reset_index(name="n_series")
    )
    summary.to_csv(metadata_dir / "ct_head_chest_selection_summary.csv", index=False)


def download_one(
    *,
    gen3_client: Path,
    profile: str,
    object_id: str,
    output_dir: Path,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(gen3_client),
        "download-single",
        f"--profile={profile}",
        f"--guid={object_id}",
        f"--download-path={output_dir}",
        "--no-prompt",
        "--skip-completed",
    ]
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        logger.warning("Download failed for %s", object_id)
        if result.stdout:
            logger.warning("stdout: %s", result.stdout.strip())
        if result.stderr:
            logger.warning("stderr: %s", result.stderr.strip())
        return False
    return True


def download_selected(
    selected: pd.DataFrame,
    *,
    gen3_client: Path,
    profile: str,
    raw_dir: Path,
) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        source_class = _safe_text(row["source_class"])
        object_id = _safe_text(row["object_id"])
        ok = download_one(
            gen3_client=gen3_client,
            profile=profile,
            object_id=object_id,
            output_dir=raw_dir / source_class,
        )
        out = row.to_dict()
        out["download_ok"] = ok
        rows.append(out)
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--cred", type=Path, default=DEFAULT_CRED)
    parser.add_argument("--gen3-client", type=Path, default=DEFAULT_GEN3_CLIENT)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument(
        "--selected-csv",
        type=Path,
        default=None,
        help=(
            "Existing selected cohort CSV to reuse. Defaults to "
            "<metadata-dir>/ct_head_chest_selected.csv."
        ),
    )
    parser.add_argument(
        "--use-existing-selection",
        action="store_true",
        help="Skip the MIDRC metadata query and reuse --selected-csv.",
    )
    parser.add_argument("--target-per-class", type=int, default=100)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument(
        "--exclude-object-id",
        action="append",
        default=[],
        help=(
            "Object ID to exclude from selection. Repeat for multiple known "
            "bad or incompatible objects."
        ),
    )
    parser.add_argument(
        "--exclude-series-uid",
        action="append",
        default=[],
        help=(
            "Series UID to exclude from selection. Repeat for multiple known "
            "bad or incompatible series."
        ),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    selected_csv = (
        args.selected_csv
        if args.selected_csv is not None
        else args.metadata_dir / "ct_head_chest_selected.csv"
    )

    if not args.use_existing_selection and not args.cred.exists():
        raise FileNotFoundError(f"credentials file does not exist: {args.cred}")
    if args.download and not args.gen3_client.exists():
        raise FileNotFoundError(f"gen3-client does not exist: {args.gen3_client}")

    if args.use_existing_selection:
        if not selected_csv.exists():
            raise FileNotFoundError(f"selected cohort CSV does not exist: {selected_csv}")
        selected = pd.read_csv(selected_csv)
        candidates = selected.copy()
        records = []
    else:
        auth = Gen3Auth(args.api, refresh_file=str(args.cred))
        query = Gen3Query(auth)
        records = query_ct_records(
            query,
            page_size=args.page_size,
            max_records=args.max_records,
        )
        candidates, selected = build_selected_cohort(
            records,
            target_per_class=args.target_per_class,
            exclude_object_ids=set(args.exclude_object_id),
            exclude_series_uids=set(args.exclude_series_uid),
        )
        write_metadata_outputs(candidates, selected, args.metadata_dir)

    print()
    print("=" * 72)
    print(f"  CT metadata records queried : {len(records)}")
    print(f"  used existing selection     : {args.use_existing_selection}")
    print(f"  labeled CT candidates      : {len(candidates)}")
    print(f"  selected series            : {len(selected)}")
    print(f"  metadata dir               : {args.metadata_dir}")
    print("=" * 72)
    if not selected.empty:
        print(
            selected.groupby(["source_class", "body_region"], dropna=False)
            .size()
            .reset_index(name="n_series")
            .to_string(index=False)
        )

    if args.download:
        downloaded = download_selected(
            selected,
            gen3_client=args.gen3_client,
            profile=args.profile,
            raw_dir=args.raw_dir,
        )
        downloaded.to_csv(args.metadata_dir / "ct_head_chest_downloads.csv", index=False)
        print()
        print("=" * 72)
        print(f"  download attempts : {len(downloaded)}")
        print(f"  download ok       : {int(downloaded['download_ok'].sum())}")
        print(f"  raw dir           : {args.raw_dir}")
        print("=" * 72)
    else:
        print()
        print("Metadata-only run complete. Add --download to fetch selected ZIPs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
