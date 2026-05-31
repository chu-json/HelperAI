# HelperAI

A local, freely-licensed QA / classification pilot for the Langlotz Lab / RSNA / Cornell
Radiology project. It is **not** a diagnostic tool.

**Goal:** a safety gate that checks whether a DICOM series is the expected kind of imaging
(modality, body part, view) before it is passed to a downstream AI system.

**Phase 1 model:** body-part classifier (CHEST / ABDOMEN / HEAD / SPINE / EXTREMITY)
with view sub-classification for X-rays (AP / PA / LATERAL), trained on a ResNet
fine-tuned from ImageNet weights.

---

## Pipeline overview

```
Step 1  query_midrc_bulk.py       ← query MIDRC, write per-class manifests
Step 2  gen3-client                ← bulk download ZIPs from MIDRC
Step 3  extract_midrc_zips.py     ← unzip into data/extracted/
Step 4  build_dataset.py          ← filter derived series, assign labels, split
Step 5  export_images.py          ← DICOM → 224×224 RGB PNG for ResNet input
```

---

## Repo layout

```
HelperAI/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_midrc_tiny_download.ipynb   # auth + tiny MIDRC pull (Phase 0)
│   ├── 02_inspect_dicoms.ipynb        # series-level grouping
│   └── 03_visual_inspect.ipynb        # visual sanity check
├── scripts/
│   ├── query_midrc_bulk.py            # Step 1 — bulk MIDRC query + manifest
│   ├── extract_midrc_zips.py          # Step 3 — unzip downloaded series
│   ├── inspect_dicoms.py              # (utility) scan + summarise DICOM headers
│   ├── build_dataset.py               # Step 4 — filter, label, split
│   └── export_images.py               # Step 5 — DICOM → PNG
├── data/                              # gitignored
│   ├── raw/          <class>/         # downloaded ZIPs
│   ├── extracted/    <class>/         # unzipped DICOM series
│   ├── images/       train|val|test/<LABEL>/<uid>.png
│   ├── metadata/                      # per-class query result CSVs
│   └── manifests/                     # gen3-client manifest JSON files
└── outputs/
    ├── sample_series_summary.csv      # safe-to-commit aggregate (no PatientID)
    ├── dataset_manifest.csv           # series manifest with labels + splits
    └── image_manifest.csv             # image manifest with PNG paths
```

---

## Setup

```bash
pip install -r requirements.txt
```

Configure the Gen3 client once (use the `credentials.json` from the MIDRC portal):

```bash
/Applications/gen3-client configure \
  --profile=midrc \
  --cred=/Users/jason/Downloads/credentials.json \
  --apiendpoint=https://data.midrc.org/

/Applications/gen3-client auth --profile=midrc
```

---

## Step 1 — Query MIDRC (bulk)

```bash
# Dry run: verify connectivity and print record counts
python scripts/query_midrc_bulk.py --max-per-class 1000 --dry-run

# Full query: write manifests to data/manifests/
python scripts/query_midrc_bulk.py --max-per-class 1000

# Specific classes only
python scripts/query_midrc_bulk.py --classes chest_xray head_ct --max-per-class 500
```

Available classes: `chest_xray`, `abdomen_xray`, `head_ct`

---

## Step 2 — Download ZIPs via gen3-client

The query script prints the exact commands. Example:

```bash
/Applications/gen3-client download-multiple-files \
    --profile=midrc \
    --manifest=data/manifests/chest_xray_manifest.json \
    --download-path=data/raw/chest_xray/ \
    --numparallel=5 \
    --skip-completed
```

---

## Step 3 — Extract ZIPs

```bash
python scripts/extract_midrc_zips.py
# or with options:
python scripts/extract_midrc_zips.py --input data/raw --output data/extracted
```

---

## Step 4 — Build dataset manifest

Reads all extracted DICOMs, filters derived / non-imaging series, assigns
body-part and view labels from DICOM metadata, and creates a patient-level
stratified 70/15/15 train/val/test split.

```bash
python scripts/build_dataset.py
# or:
python scripts/build_dataset.py \
    --input data/extracted \
    --output outputs/dataset_manifest.csv \
    --drop-unknown          # drop series whose body part cannot be resolved
```

Output: `outputs/dataset_manifest.csv` with columns:
- `series_uid`, `study_uid`, `patient_id`
- `modality`, `body_part_label` (CHEST / ABDOMEN / HEAD / SPINE / EXTREMITY)
- `view_label` (AP / PA / LATERAL / OBLIQUE / "" for CT/MR)
- `n_files`, `representative_dcm`, `split`

### Filtering logic (derived series)

Series are excluded when:
- `ImageType` starts with `DERIVED` (segmentation, subtraction, MIP, etc.)
- `SeriesDescription` matches patterns: `Segmentation`, `BrainSegmentation`,
  `Scout`, `Localizer`, `SR`, `KOS`, `Dose Report`, etc.

This addresses the concern about brain-segmentation derived series appearing in
the head CT class — only raw ORIGINAL CT slices are kept.

---

## Step 5 — Export images

Converts each series to a single representative 224×224 RGB PNG:

```bash
python scripts/export_images.py
# Options:
python scripts/export_images.py --size 224 --ct-window brain --workers 8
```

**CT windowing presets** (`--ct-window`):

| Preset       | Center | Width |
|-------------|--------|-------|
| brain        |    40  |    80 |
| soft_tissue  |    50  |   400 |
| lung         |  -600  |  1500 |
| bone         |   400  |  1800 |
| chest        |    40  |   400 |

Output:
```
data/images/
    train/CHEST/   HEAD/   ABDOMEN/   SPINE/   EXTREMITY/
    val/  ...
    test/ ...
```

Compatible with `torchvision.datasets.ImageFolder` for direct ResNet training.

---

## Label strategy (weak ground truth)

Labels are derived from DICOM metadata tags in priority order:

1. `BodyPartExamined` → normalized to canonical label
2. `StudyDescription` → keyword search
3. Source-class folder name (`chest_xray` → CHEST, etc.)

View labels (X-ray only):
1. `ViewPosition` tag (most reliable: AP, PA, LL, RL)
2. `SeriesDescription` keyword search

These are **weak labels** — the notes call for:
- Human review of false positives/negatives
- LLM-assisted fuzzy matching for ambiguous StudyDescription strings
- Corrected ground truth labels fed back into training (see future work)

---

## Safety rules (do not violate)

- Never commit `credentials.json`.
- Never commit anything under `data/` (DICOMs, metadata CSVs, manifests, images).
- Never commit `outputs/dicom_file_metadata.csv` or `outputs/image_manifest.csv`
  (they may contain patient-level identifiers).
- Only `outputs/sample_series_summary.csv` and `outputs/dataset_manifest.csv`
  (aggregated, no PatientID column) are intended to be committable.
- Do not commit trained model weights containing embedded training data.
