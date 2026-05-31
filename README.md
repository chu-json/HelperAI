# HelperAI

A local, freely licensed QA / classification pilot for the Langlotz Lab / RSNA /
Cornell Radiology project. It is **not** a diagnostic tool.

**Goal:** build a safety gate that checks whether a DICOM series is the expected
kind of imaging (modality, body part, view) before it is passed to a downstream
AI system.

## Current Scope

This repo now contains two complementary local experimentation paths:

1. **General image-export pipeline**: query MIDRC classes, download ZIPs,
   extract DICOMs, build a dataset manifest, and export 224x224 PNGs compatible
   with `torchvision.datasets.ImageFolder`.
2. **CT mosaic pipeline**: query a balanced head/chest CT cohort, build one
   series-level mosaic PNG per CT series, train a ResNet-18 classifier on those
   mosaics, and analyze model results.

All labels are metadata-derived weak labels, not radiologist-verified clinical
ground truth. The repo does not upload, share, or commit raw imaging data.

## Repo Layout

```text
HelperAI/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_midrc_tiny_download.ipynb        # auth + tiny MIDRC pull
│   ├── 02_extract_and_inspect_dicoms.ipynb # unzip + series grouping
│   ├── 02_inspect_dicoms.ipynb             # alternate inspection notebook
│   ├── 03_visual_inspect.ipynb             # raw DICOM visual QA
│   ├── 04_create_mosaics.ipynb             # CT mosaic generation QA
│   └── 05_analyze_model_results.ipynb      # reusable model results analysis
├── scripts/
│   ├── query_midrc_bulk.py                 # bulk MIDRC query + manifests
│   ├── extract_midrc_zips.py               # unzip downloaded series
│   ├── inspect_dicoms.py                   # DICOM header summary utility
│   ├── build_dataset.py                    # filter, label, split extracted DICOMs
│   ├── export_images.py                    # DICOM series -> PNG exports
│   ├── download_midrc_ct_cohort.py         # CT head/chest cohort selector
│   ├── build_midrc_mosaic_dataset.py       # download one series -> mosaic -> cleanup
│   ├── create_series_mosaics.py            # reusable CT mosaic generator
│   └── train_head_chest_resnet18.py        # train ResNet-18 on mosaic PNGs
├── data/                                   # gitignored
│   ├── raw/                                # downloaded ZIPs / DICOMs
│   ├── extracted/                          # unzipped DICOM series
│   ├── images/                             # exported ImageFolder-style PNGs
│   ├── metadata/                           # query result CSVs
│   ├── manifests/                          # Gen3 manifests
│   └── mosaics/                            # generated CT mosaic PNGs
└── outputs/
    ├── sample_series_summary.csv           # safe-to-commit aggregate summary
    ├── dataset_manifest.csv                # general series labels + splits
    ├── image_manifest.csv                  # exported PNG paths
    └── mosaic_manifest.csv                 # CT mosaic labels + paths
```

## Setup

```bash
pip install -r requirements.txt
```

Configure the Gen3 client once using the `credentials.json` from the MIDRC
portal:

```bash
/Applications/gen3-client configure \
  --profile=midrc \
  --cred=~/Downloads/credentials.json \
  --apiendpoint=https://data.midrc.org/

/Applications/gen3-client auth --profile=midrc
```

## Path A: General Image-Export Dataset

Use this path when you want a broader body-part / view dataset exported as PNGs
that can be loaded directly with `torchvision.datasets.ImageFolder`.

### Step 1 - Query MIDRC

```bash
# Dry run: verify connectivity and print record counts
python scripts/query_midrc_bulk.py --max-per-class 1000 --dry-run

# Full query: write manifests under data/manifests/
python scripts/query_midrc_bulk.py --max-per-class 1000

# Specific classes only
python scripts/query_midrc_bulk.py --classes chest_xray head_ct --max-per-class 500
```

Available classes include `chest_xray`, `abdomen_xray`, and `head_ct`.

### Step 2 - Download ZIPs via gen3-client

The query script prints the exact commands. Example:

```bash
/Applications/gen3-client download-multiple-files \
  --profile=midrc \
  --manifest=data/manifests/chest_xray_manifest.json \
  --download-path=data/raw/chest_xray/ \
  --numparallel=5 \
  --skip-completed
```

### Step 3 - Extract ZIPs

```bash
python scripts/extract_midrc_zips.py
# or with options:
python scripts/extract_midrc_zips.py --input data/raw --output data/extracted
```

### Step 4 - Build Dataset Manifest

Reads extracted DICOMs, filters derived / non-imaging series, assigns body-part
and view labels from DICOM metadata, and creates a patient-level stratified
70/15/15 train/val/test split.

```bash
python scripts/build_dataset.py
# or:
python scripts/build_dataset.py \
  --input data/extracted \
  --output outputs/dataset_manifest.csv \
  --drop-unknown
```

Output: `outputs/dataset_manifest.csv` with columns such as:

- `series_uid`, `study_uid`, `patient_id`
- `modality`, `body_part_label`, `view_label`
- `n_files`, `representative_dcm`, `split`

Derived series are excluded when `ImageType` starts with `DERIVED` or
`SeriesDescription` matches patterns such as `Segmentation`, `Scout`,
`Localizer`, `SR`, `KOS`, or `Dose Report`.

### Step 5 - Export Images

Converts each series to a single representative 224x224 RGB PNG:

```bash
python scripts/export_images.py
# Options:
python scripts/export_images.py --size 224 --ct-window brain --workers 8
```

Output:

```text
data/images/
  train/CHEST/ HEAD/ ABDOMEN/ SPINE/ EXTREMITY/
  val/...
  test/...
```

## Path B: CT Mosaic Head/Chest Classifier

Use this path for the current CT-only series mosaic classifier.

### Step 1 - Optional Notebook Exploration

```bash
jupyter notebook notebooks/01_midrc_tiny_download.ipynb
jupyter notebook notebooks/02_extract_and_inspect_dicoms.ipynb
jupyter notebook notebooks/03_visual_inspect.ipynb
jupyter notebook notebooks/04_create_mosaics.ipynb
jupyter notebook notebooks/05_analyze_model_results.ipynb
```

### Step 2 - Query a Balanced CT Head/Chest Cohort

Query 100 head CT and 100 chest CT DICOM series. This writes metadata CSVs under
`data/metadata/` and does not download pixel data:

```bash
python scripts/download_midrc_ct_cohort.py --target-per-class 100
```

The selector keeps DICOM CT records only (`data_type=DICOM`, `data_format=DCM`)
and labels body region conservatively from metadata text. If MIDRC returns a
known bad object, exclude it repeatably:

```bash
python scripts/download_midrc_ct_cohort.py \
  --target-per-class 100 \
  --exclude-object-id dg.MD1R/example-bad-object-id \
  --exclude-series-uid example.bad.series.uid
```

### Step 3 - Build CT Mosaic PNGs

Build mosaics one series at a time. The script downloads a series ZIP into a
temporary work folder, extracts it, writes one mosaic PNG, appends the manifest,
and cleans up the raw ZIP/DICOM files before moving to the next series:

```bash
python scripts/build_midrc_mosaic_dataset.py
```

Expected durable local outputs:

- `data/mosaics/<SeriesInstanceUID>.png` - generated images, gitignored.
- `outputs/mosaic_manifest.csv` - safe-to-commit index with labels and paths.
- `data/metadata/ct_mosaic_build_failures.csv` - failures to review/retry.

Verify the balanced dataset:

```bash
python - <<'PY'
import pandas as pd
m = pd.read_csv("outputs/mosaic_manifest.csv")
print(len(m))
print(m["body_region"].value_counts())
PY
```

### Step 4 - Train ResNet-18 on Mosaics

Train the binary classifier from the manifest. The default uses ImageNet
pretrained ResNet-18 weights, resizes mosaics to 224x224, and creates
deterministic stratified train/validation/test splits:

```bash
python scripts/train_head_chest_resnet18.py --epochs 10
```

For a quick smoke test without downloading pretrained weights:

```bash
python scripts/train_head_chest_resnet18.py --epochs 1 --no-pretrained
```

Training artifacts are written under `outputs/resnet18_head_chest/` and are
gitignored:

- `config.json` - training configuration for the run.
- `label_mapping.json` - numeric class mapping, currently `head -> 0` and
  `chest -> 1`.
- `splits.csv` - train/validation/test assignment for each series.
- `history.csv` - per-epoch train loss and validation metrics.
- `best_resnet18.pt` - weights from the epoch with best validation balanced
  accuracy.
- `final_resnet18.pt` - weights after the final training epoch.
- `test_metrics.json` - test accuracy, balanced accuracy, classification report,
  and confusion matrix.
- `test_predictions.csv` - per-series test predictions.

### Step 5 - Analyze Model Results

Open the reusable analysis notebook after training:

```bash
jupyter notebook notebooks/05_analyze_model_results.ipynb
```

The notebook defaults to `outputs/resnet18_head_chest/` and analyzes
`test_metrics.json`, `test_predictions.csv`, `splits.csv`, `history.csv`, and
`outputs/mosaic_manifest.csv`. To analyze a future model run or expanded class
set, change `RESULTS_DIR` in the first notebook code cell.

It includes:

- dataset and split balance checks
- training curves
- classification report
- raw and normalized confusion matrices
- prediction review
- mosaic previews for mistakes and sampled correct predictions
- notes on weak labels, series-level splits, and pilot limitations

## Label Strategy

Labels are derived from DICOM and MIDRC metadata, in priority order depending on
the pipeline:

1. Structured tags such as `BodyPartExamined` and `ViewPosition`
2. Study / series description keyword search
3. Source-class folder or cohort metadata

These are **weak labels**. Future work should include human review of false
positives/negatives, better fuzzy matching for ambiguous descriptions, and
corrected labels fed back into training.

## Safety Rules

- Never commit `credentials.json`.
- Never commit anything under `data/` (DICOMs, metadata CSVs, manifests, images,
  or mosaics).
- Never commit `outputs/dicom_file_metadata.csv` or `outputs/image_manifest.csv`
  if they contain patient-level identifiers.
- Do not commit trained model weights (`*.pt`, `*.pth`) unless there is an
  explicit reason and review.
- `outputs/sample_series_summary.csv`, `outputs/dataset_manifest.csv`, and
  `outputs/mosaic_manifest.csv` are intended to be committable only when they
  contain aggregate metadata and no patient identifiers.
- Training metrics and split CSVs under `outputs/resnet18_head_chest/` are local
  experiment artifacts by default.
- Start tiny. The first success criterion is "a nonzero number of `.dcm` files
  downloaded", not a large cohort.
