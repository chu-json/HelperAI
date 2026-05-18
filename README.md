# HelperAI

A small, local, freely-licensed pilot tool for the Langlotz Lab / RSNA / Cornell
Radiology pilot. It is **not** a diagnostic tool.

The long-term goal is a QA / classification "safety gate" that checks whether a
DICOM series is the expected kind of imaging (modality, body part, view) before
it is passed to a downstream AI system.

## Current scope (Phase 0)

This repo currently does **only** the following:

1. Downloads a *tiny* sample of DICOMs from MIDRC via Gen3 (5-10 files / class).
2. Inspects DICOM headers and groups files by `SeriesInstanceUID`.

It does **not** train any model yet. It does not upload, share, or commit any
imaging data.

## Repo layout

```
HelperAI/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_midrc_tiny_download.ipynb   # auth + tiny MIDRC pull
│   └── 02_inspect_dicoms.ipynb        # series-level grouping
├── scripts/
│   └── inspect_dicoms.py              # CLI version of notebook 02
├── data/                              # gitignored
│   ├── raw/                           # downloaded DICOMs (gitignored)
│   ├── metadata/                      # query result CSVs (gitignored)
│   └── manifests/                     # Gen3 manifests (gitignored)
└── outputs/
    └── sample_series_summary.csv      # safe-to-commit aggregate summary
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
  --cred=/Users/jason/Downloads/credentials.json \
  --apiendpoint=https://data.midrc.org/

/Applications/gen3-client auth --profile=midrc
```

## Run

```bash
jupyter notebook notebooks/01_midrc_tiny_download.ipynb
jupyter notebook notebooks/02_inspect_dicoms.ipynb
```

Or use the CLI inspector:

```bash
python scripts/inspect_dicoms.py \
  --input  data/raw \
  --output outputs/sample_series_summary.csv
```

## Safety rules (do not violate)

- Never commit `credentials.json`.
- Never commit anything under `data/` (DICOMs, metadata CSVs, manifests).
- Never commit `outputs/dicom_file_metadata.csv` (it contains per-file IDs).
- Only `outputs/sample_series_summary.csv` (aggregated, no PatientID) is meant
  to be committable.
- Start tiny. The first success criterion is "a nonzero number of `.dcm`
  files downloaded", not a large cohort.
