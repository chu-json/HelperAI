# HelperAI

A small, local, freely-licensed pilot tool for the Langlotz Lab / RSNA / Cornell
Radiology pilot. It is **not** a diagnostic tool.

The long-term goal is a QA / classification "safety gate" that checks whether a
DICOM series is the expected kind of imaging (modality, body part, view) before
it is passed to a downstream AI system.

## Current scope

This repo currently supports two local experimental phases:

1. **Phase 0: data preparation** - query MIDRC CT metadata, select a balanced
   head/chest cohort, and build CT series-level mosaic PNGs.
2. **Phase 1: model training** - train a ResNet-18 classifier that predicts
   whether a CT mosaic is `head` or `chest`.

It does not upload, share, or commit imaging data. The labels are weak labels
derived from MIDRC metadata, not clinical ground truth.

## Repo layout

```
HelperAI/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_midrc_tiny_download.ipynb   # auth + tiny MIDRC pull
│   ├── 02_extract_and_inspect_dicoms.ipynb # unzip + series grouping
│   ├── 03_visual_inspect.ipynb        # raw DICOM visual QA
│   ├── 04_create_mosaics.ipynb        # CT mosaic generation QA
│   └── 05_analyze_model_results.ipynb # reusable model results analysis
├── scripts/
│   ├── extract_midrc_zips.py          # extract MIDRC ZIPs
│   ├── inspect_dicoms.py              # series-level DICOM summary
│   ├── create_series_mosaics.py       # CT series mosaic generator
│   ├── download_midrc_ct_cohort.py    # CT head/chest MIDRC cohort selector
│   ├── build_midrc_mosaic_dataset.py  # download one series -> mosaic -> cleanup
│   └── train_head_chest_resnet18.py   # train ResNet-18 on mosaic PNGs
├── data/                              # gitignored
│   ├── raw/                           # downloaded DICOMs (gitignored)
│   ├── extracted/                     # unzipped DICOMs (gitignored)
│   ├── metadata/                      # query result CSVs (gitignored)
│   ├── manifests/                     # Gen3 manifests (gitignored)
│   └── mosaics/                       # generated CT PNG mosaics (gitignored)
└── outputs/
    ├── sample_series_summary.csv      # safe-to-commit aggregate summary
    ├── mosaic_manifest.csv            # safe-to-commit mosaic manifest
    └── resnet18_head_chest/           # local training artifacts (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

Configure the Gen3 client once (uses the credentials.json you downloaded from
the MIDRC portal):

```bash
/Applications/gen3-client configure \
  --profile=midrc \
  --cred=~/Downloads/credentials.json \
  --apiendpoint=https://data.midrc.org/

/Applications/gen3-client auth --profile=midrc
```

## Run the Full V1 Pipeline

### 1. Optional tiny notebook path

```bash
jupyter notebook notebooks/01_midrc_tiny_download.ipynb
jupyter notebook notebooks/02_extract_and_inspect_dicoms.ipynb
jupyter notebook notebooks/03_visual_inspect.ipynb
jupyter notebook notebooks/04_create_mosaics.ipynb
jupyter notebook notebooks/05_analyze_model_results.ipynb
```

Use this path for manual exploration and visual QA. For the balanced CT
head/chest classifier dataset, prefer the script path below.

### 2. Query a balanced CT head/chest cohort

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

### 3. Build CT mosaic PNGs

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

### 4. Train ResNet-18 on the mosaics

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

### 5. Analyze model results

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

### 6. Optional standalone DICOM inspection / mosaic creation

If you already have extracted DICOMs and want to inspect or build mosaics
outside the V1 one-series-at-a-time builder, use:

```bash
python scripts/inspect_dicoms.py \
  --input  data/extracted \
  --output outputs/sample_series_summary.csv
```

Create CT series-level mosaics:

```bash
python scripts/create_series_mosaics.py
```

Pipeline overview:

```text
MIDRC query -> selected cohort CSV -> per-series ZIP download -> mosaic PNGs
  -> mosaic manifest -> model train/val/test split -> metrics/checkpoints
  -> reusable results analysis notebook
```

## Safety rules (do not violate)

- Never commit `credentials.json`.
- Never commit anything under `data/` (DICOMs, metadata CSVs, manifests).
- Never commit `data/mosaics/` (mosaics are derived image data).
- Never commit `outputs/dicom_file_metadata.csv` (it contains per-file IDs).
- Never commit model checkpoints (`*.pt`, `*.pth`) unless there is an explicit
  reason and review.
- Only `outputs/sample_series_summary.csv` (aggregated, no PatientID) is meant
  to be committable.
- `outputs/mosaic_manifest.csv` is meant to be committable; it records safe
  aggregate mosaic metadata and relative paths, not pixel data.
- Training metrics and split CSVs under `outputs/resnet18_head_chest/` are local
  experiment artifacts by default.
- Start tiny. The first success criterion is "a nonzero number of `.dcm`
  files downloaded", not a large cohort.
