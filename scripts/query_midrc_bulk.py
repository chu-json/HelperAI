#!/usr/bin/env python3
"""Query MIDRC for multiple body-part / modality classes and write a gen3-client manifest.

Pipeline position:
    [THIS SCRIPT] → gen3-client download-multiple-files → extract_midrc_zips.py → ...

Usage:
    # Dry run — print record counts, write CSVs, do NOT download
    python scripts/query_midrc_bulk.py --max-per-class 1000 --dry-run

    # Query + write manifest, then print the download command
    python scripts/query_midrc_bulk.py --max-per-class 1000

    # Also shell out gen3-client immediately
    python scripts/query_midrc_bulk.py --max-per-class 1000 --auto-download

After running (without --auto-download), execute:
    gen3-client download-multiple-files \\
        --profile=midrc \\
        --manifest=data/manifests/<label>_manifest.json \\
        --download-path=data/raw/<label>/ \\
        --numparallel=5 \\
        --skip-completed

Design notes:
- Uses MIDRC's flat-query API via Gen3 Python SDK (raw_data_download / query fallback).
- Paginates with offset so we can retrieve more than 2000 records per class.
- Filters at query time by modality (reliable) and optionally body_part_examined or
  study_description (less reliable — used as a coarse pre-filter only).
- The downstream build_dataset.py does the strict filtering on actual DICOM headers.
- Brain/head CT: intentionally does NOT restrict to "BrainSegmentation" study desc —
  we want raw CT series; build_dataset.py will filter out derived seg series.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("query_midrc_bulk")

# ---------------------------------------------------------------------------
# Paths / auth
# ---------------------------------------------------------------------------
API = "https://data.midrc.org"
CRED = "/Users/jason/Downloads/credentials.json"
GEN3_CLIENT = "/Applications/gen3-client"
PROFILE = "midrc"

REPO_ROOT = Path("/Users/jason/HelperAI")
META_DIR = REPO_ROOT / "data" / "metadata"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
RAW_DIR = REPO_ROOT / "data" / "raw"

# Fields to request from the MIDRC flat-query API.
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

# ---------------------------------------------------------------------------
# Cohort configuration
# ---------------------------------------------------------------------------
# Each entry defines one output class (directory label + manifest file).
# `filter` is passed directly to MIDRC; you can use AND/IN/=eq combinations.
# Keep filters broad here — strict per-series filtering happens in build_dataset.py.
#
# Note on head_ct: body_part_examined is often empty in MIDRC CT records, so we
# rely on study_description whitelists instead.  The list below is deliberately
# broad (many study description variants across institutions in MIDRC).

COHORT_CONFIG: dict[str, dict] = {
    "chest_xray": {
        "description": "Chest radiographs (DX / CR)",
        "filter": {
            "AND": [
                {"IN": {"modality": ["CR", "DX"]}},
                {
                    "IN": {
                        "body_part_examined": [
                            "CHEST",
                            "PORT CHEST",
                            "THORAX",
                            "CHST",
                            "LUNG",
                            "LUNGS",
                            "RIBS",
                        ]
                    }
                },
            ]
        },
    },
    "abdomen_xray": {
        "description": "Abdominal radiographs (DX / CR)",
        "filter": {
            "AND": [
                {"IN": {"modality": ["CR", "DX"]}},
                {
                    "IN": {
                        "body_part_examined": [
                            "ABDOMEN",
                            "ABD",
                            "ABDO",
                            "ABDOMINAL",
                            "ABDO_PELVIS",
                            "ABDPELV",
                        ]
                    }
                },
            ]
        },
    },
    "head_ct": {
        "description": "Head / brain CT (non-contrast, raw series only)",
        "filter": {
            "AND": [
                {"IN": {"modality": ["CT"]}},
                {
                    "IN": {
                        "study_description": [
                            # MIDRC / IU variants
                            "BRAIN W/O CONTRAST (CT)-CS",
                            "CT HEAD W/O CONTRAST",
                            "CT HEAD WITHOUT CONTRAST",
                            "CT BRAIN W/O CONTRAST",
                            "CT BRAIN WITHOUT CONTRAST",
                            "CT BRAIN W CONTRAST",
                            "HEAD CT W/O CONTRAST",
                            "HEAD CT WITHOUT CONTRAST",
                            "CT OF THE HEAD",
                            "CT HEAD",
                            "BRAIN CT",
                            "HEAD W/O CONTRAST",
                        ]
                    }
                },
                # Exclude derived segmentation series at query time.
                # ~98.5% of MIDRC head CT series records are BrainSegmentation;
                # filtering here avoids downloading thousands of unusable ZIPs.
                {"!=": {"series_description": "BrainSegmentation"}},
            ]
        },
    },
}

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------
PAGE_SIZE = 500  # records per API page (conservative; MIDRC allows ~2000)
RETRY_WAIT = 5   # seconds between retries on transient errors
MAX_RETRIES = 3


def _query_page(
    query_obj: Any,
    filter_obj: dict,
    first: int,
    offset: int,
) -> list[dict]:
    """Fetch one page of records from MIDRC. Returns empty list on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            if hasattr(query_obj, "raw_data_download"):
                records = query_obj.raw_data_download(
                    data_type="data_file",
                    fields=RETURN_FIELDS,
                    filter_object=filter_obj,
                    first=first,
                    offset=offset,
                )
                return records or []
            # Fallback: older SDK versions use query()
            fields_str = " ".join(RETURN_FIELDS)
            gql = (
                f'{{\n  data_file(first: {first}, offset: {offset},'
                f' filter: {json.dumps(filter_obj)}) {{\n    {fields_str}\n  }}\n}}'
            )
            res = query_obj.query(gql)
            return (res.get("data", {}) or {}).get("data_file", []) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("Page query failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
    return []


def query_class(
    query_obj: Any,
    label: str,
    config: dict,
    max_records: int,
    *,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Paginate through all records for one class and return a DataFrame."""
    filter_obj = config["filter"]
    logger.info("[%s] querying MIDRC (max=%d) …", label, max_records)

    if dry_run:
        # Fetch one small page just to verify connectivity + print columns.
        sample = _query_page(query_obj, filter_obj, first=5, offset=0)
        if sample:
            cols = list(sample[0].keys())
            logger.info("[%s] dry-run: got %d sample records. Columns: %s", label, len(sample), cols)
        else:
            logger.warning("[%s] dry-run: got 0 records — check filter or auth.", label)
        return pd.DataFrame(sample)

    all_records: list[dict] = []
    offset = 0
    while len(all_records) < max_records:
        want = min(PAGE_SIZE, max_records - len(all_records))
        batch = _query_page(query_obj, filter_obj, first=want, offset=offset)
        if not batch:
            break
        all_records.extend(batch)
        logger.info("[%s] fetched %d / %d records (offset=%d)", label, len(all_records), max_records, offset)
        if len(batch) < want:
            break  # reached end of results
        offset += len(batch)

    df = pd.DataFrame(all_records)
    logger.info("[%s] total records: %d", label, len(df))
    return df


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def write_manifest(df: pd.DataFrame, label: str) -> Path | None:
    """Write a gen3-client compatible JSON manifest for one class."""
    if df.empty or "object_id" not in df.columns:
        logger.warning("[%s] no object_id column — skipping manifest.", label)
        return None

    records = [
        {
            "object_id": row["object_id"],
            **({"file_name": row["file_name"]} if "file_name" in df.columns else {}),
            **({"file_size": int(row["file_size"])} if "file_size" in df.columns and pd.notna(row.get("file_size")) else {}),
        }
        for _, row in df.iterrows()
        if row.get("object_id")
    ]

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{label}_manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2))
    logger.info("[%s] wrote manifest (%d GUIDs) → %s", label, len(records), manifest_path)
    return manifest_path


def download_class(label: str, manifest_path: Path, num_parallel: int = 5) -> bool:
    """Shell out to gen3-client download-multiple for one class."""
    dest = RAW_DIR / label
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        GEN3_CLIENT,
        "download-multiple",
        f"--profile={PROFILE}",
        f"--manifest={manifest_path}",
        f"--download-path={dest}",
        f"--numparallel={num_parallel}",
        "--no-prompt",
        "--skip-completed",
    ]
    logger.info("[%s] running: %s", label, " ".join(cmd))
    try:
        result = subprocess.run(cmd, text=True, timeout=3600)
        if result.returncode != 0:
            logger.error("[%s] gen3-client exited %d", label, result.returncode)
            return False
        logger.info("[%s] download completed.", label)
        return True
    except subprocess.TimeoutExpired:
        logger.error("[%s] download timed out after 1 hour.", label)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] download error: %s", label, exc)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=list(COHORT_CONFIG.keys()),
        choices=list(COHORT_CONFIG.keys()),
        metavar="CLASS",
        help=f"Which classes to query (default: all). Options: {list(COHORT_CONFIG.keys())}",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=1000,
        help="Max records (series) to retrieve per class (default: 1000).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch only a 5-record sample per class, print columns, do not write manifests.",
    )
    p.add_argument(
        "--auto-download",
        action="store_true",
        help="After querying, shell out gen3-client to download immediately.",
    )
    p.add_argument(
        "--num-parallel",
        type=int,
        default=5,
        help="--numparallel passed to gen3-client (default: 5).",
    )
    p.add_argument(
        "--cred",
        default=CRED,
        help=f"Path to credentials.json (default: {CRED}).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Connect to MIDRC.
    try:
        from gen3.auth import Gen3Auth
        from gen3.query import Gen3Query
    except ImportError:
        logger.error("gen3 SDK not installed. Run: pip install -r requirements.txt")
        return 1

    auth = Gen3Auth(API, refresh_file=args.cred)
    query_obj = Gen3Query(auth)
    logger.info("Connected to %s", API)

    META_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for label in args.classes:
        config = COHORT_CONFIG[label]
        logger.info("[%s] %s", label, config["description"])

        df = query_class(
            query_obj,
            label,
            config,
            max_records=args.max_per_class,
            dry_run=args.dry_run,
        )

        # Save per-class metadata CSV (gitignored — may contain case IDs).
        csv_path = META_DIR / f"{label}_records.csv"
        df.to_csv(csv_path, index=False)
        logger.info("[%s] saved %d rows → %s", label, len(df), csv_path)
        results[label] = df

    if args.dry_run:
        print("\nDry-run complete. Counts per class:")
        for label, df in results.items():
            print(f"  {label:20s} {len(df)} records (sample)")
        return 0

    # Write manifests.
    manifests: dict[str, Path] = {}
    for label, df in results.items():
        mp = write_manifest(df, label)
        if mp:
            manifests[label] = mp

    # Print summary.
    print()
    print("=" * 70)
    print("  QUERY SUMMARY")
    print("=" * 70)
    for label, df in results.items():
        n_guids = df["object_id"].notna().sum() if "object_id" in df.columns else 0
        print(f"  {label:20s}  {len(df):5d} records  ({n_guids} with object_id)")
    print()
    print("  Manifests written:")
    for label, mp in manifests.items():
        print(f"    {mp}")
    print()
    print("  Next step — download each class (run these commands):")
    for label, mp in manifests.items():
        dest = RAW_DIR / label
        print(f"\n    # {COHORT_CONFIG[label]['description']}")
        print(f"    {GEN3_CLIENT} download-multiple \\")
        print(f"        --profile={PROFILE} \\")
        print(f"        --manifest={mp} \\")
        print(f"        --download-path={dest}/ \\")
        print(f"        --numparallel={args.num_parallel} \\")
        print(f"        --no-prompt \\")
        print(f"        --skip-completed")
    print()
    print("  Then run:")
    print("    python scripts/extract_midrc_zips.py")
    print("    python scripts/build_dataset.py")
    print("=" * 70)

    # Optional: download immediately.
    if args.auto_download:
        logger.info("--auto-download set; starting downloads …")
        for label, mp in manifests.items():
            download_class(label, mp, num_parallel=args.num_parallel)

    return 0


if __name__ == "__main__":
    sys.exit(main())
