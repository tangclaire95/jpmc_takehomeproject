# Income Classification + Customer Segmentation

This repository contains a complete solution for the take-home objective:
1. Train and validate a classifier that predicts whether income is `<=50K` or `>50K`.
2. Build an unsupervised segmentation model and summarize how segments differ for marketing use.

## Project files
- `train_and_segment.py` – pipeline for loading data, training/evaluating classification, and building segmentation.
- `census-bureau.columns` – ordered column names for the CPS dataset.
- `census-bureau.data` – CPS records (if present locally).
- `outputs/classification_metrics.json` – generated model evaluation metrics.
- `outputs/segmentation_summary.json` – generated segment count, quality score, and segment profiles.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python3 train_and_segment.py
```

## Notes on data loading
The script first attempts to load `census-bureau.data` from this repository using headers from `census-bureau.columns`.

If `census-bureau.data` is a Git LFS pointer (common when the large file is not pulled), the script automatically falls back to the OpenML Adult dataset so the workflow is still executable in restricted environments.

## Outputs and interpretation
After running, inspect:

### 1) Classification output
`outputs/classification_metrics.json` includes:
- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `n_rows`
- `positive_rate`

These are computed on a stratified holdout set, and sample weights are used when available.

### 2) Segmentation output
`outputs/segmentation_summary.json` includes:
- chosen number of clusters `k` (selected via best silhouette score across `k=2..8`)
- final silhouette score
- segment-level profile with weighted share, size, representative numeric averages, and top categorical values

### Marketing usage guidance
Use each segment profile to tailor campaign strategy, for example:
- **High-income-heavy segments**: premium products, loyalty upgrades, investment-focused messaging.
- **Lower-income-heavy segments**: value bundles, promotions, financing options.
- **Segments with distinct employment/education signals**: customize messaging channels and product timing.

## Reproducibility
The pipeline uses a fixed random seed (`RANDOM_STATE=42`) for train/test split, PCA, random forest, and KMeans.

