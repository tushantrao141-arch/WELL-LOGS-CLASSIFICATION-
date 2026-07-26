<<<<<<< HEAD
# Well Log Facies Classification

Predicting lithofacies from wireline log responses (GR, resistivity,
density-neutron porosity, PE) using the Council Grove Field (Kansas)
dataset, comparing a classical statistical baseline against three ML
approaches -- including a 1D convolutional network written from
scratch in NumPy, with no autograd framework, specifically to build
and check the backward pass by hand rather than trust a library to
get it right.

## Why this dataset

The original plan was to build this on Equinor's Volve field release,
but the sandbox this project was built in only has network access to
a fixed allowlist (GitHub, PyPI, a few package registries) and Volve's
data isn't hosted anywhere on that list. The Council Grove set is the
next best real option: it's real Kansas Geological Survey well data
with geologist-reviewed facies picks, compiled for Brendon Hall's 2016
SEG "facies classification using machine learning" tutorial and the
associated contest. It's smaller than Volve (8 wells, 3232 labeled
half-foot samples) and that size limit shows up directly in the
results below -- worth reading as a real constraint on what
conclusions this project can support, not a footnote.

## Problem framing

Given the 7 log curves below, predict which of 9 facies classes each
half-foot depth sample belongs to.

| Curve | What it measures |
|---|---|
| GR | Gamma ray (shale content) |
| ILD_log10 | Deep resistivity (log-scaled) |
| DeltaPHI | Neutron-density porosity difference |
| PHIND | Average neutron-density porosity |
| PE | Photoelectric factor (lithology/matrix) |
| NM_M | Nonmarine (1) / marine (2) flag |
| RELPOS | Relative position within a depositional cycle (0-1) |

Facies classes span nonmarine siliciclastics (sandstone, siltstones)
through marine shale/mudstone to a range of carbonate facies
(wackestone, dolomite, packstone, bafflestone) -- full mapping in
`src/data_loader.py::FACIES_LABELS`.

## Repository layout

```
data/raw/                 source CSVs (Council Grove training set + unlabeled test wells)
data/processed/           (empty -- feature engineering runs in-memory from src/features.py)
src/
  data_loader.py           loading, well-level train/blind splitting, facies label & adjacency maps
  features.py               per-well normalization, rolling-window features, depth-window tensors
  models.py                  RF / XGBoost factory functions + DepthWindowCNN (from-scratch NumPy 1D-CNN)
  evaluate.py                 accuracy + adjacent-facies accuracy + confusion matrix helpers
  run_lowo.py                  leave-one-well-out CV harness across all 4 models (python -m src.run_lowo)
notebooks/
  01_exploratory_data_analysis.ipynb
  02_feature_engineering.ipynb
  03_baseline_lda_random_forest.ipynb
  04_gradient_boosting_xgboost.ipynb
  05_depth_window_cnn_from_scratch.ipynb
  06_leave_one_well_out_comparison.ipynb
figures/                   PNGs saved by the notebooks
results/                   leave_one_well_out.csv, lowo_summary.csv (output of run_lowo.py)
```

## Methodology

**Validation.** Random row-splitting doesn't work for this problem --
adjacent depths in the same well are highly correlated, so a random
split lets a model see the near-neighbors of its own test rows during
training. Every model here is validated by holding out an entire well
(train on the other 7, test on the 8th), and the headline numbers
(`06`) come from running that for all 8 wells rather than trusting one
arbitrary split.

**Normalization.** Log curves are z-scored *within* each well before
pooling, to correct for calibration drift between wells/tools rather
than let a model partly learn well identity instead of facies.

**Class imbalance.** Facies 2 (nonmarine coarse siltstone) has 7.5x
the samples of facies 7 (dolomite). Every model uses either
`class_weight="balanced"` or explicit inverse-frequency sample
weights -- an unweighted model would just learn to over-predict the
majority classes.

**Metric.** Plain top-1 accuracy plus an *adjacent-facies accuracy*
that also credits predictions landing on a geologically neighboring
facies (e.g. wackestone predicted as packstone) -- the original SEG
contest used the same idea, because some of these boundaries are
genuinely ambiguous calls even from core, not model failures.

## Models

1. **LDA** -- classical discriminant analysis, the pre-ML answer to
   log-facies classification (this kind of problem was being worked
   with discriminant analysis and crossplot heuristics before "machine
   learning" was the term anyone used for it).
2. **Random Forest** -- standard tabular ML baseline.
3. **XGBoost** -- gradient boosting, usually the strongest tabular
   option; includes a small learning-rate/depth sensitivity sweep in
   `04`.
4. **DepthWindowCNN** -- a 1-conv-layer, 1D CNN over a 9-sample
   (4.5 ft) depth window, feeding a small dense head. Written by hand
   in `src/models.py`: explicit im2col forward pass, explicit
   hand-derived backward pass, mini-batch SGD with momentum,
   best-validation-epoch checkpointing. No PyTorch/TensorFlow --
   partly a sandbox constraint (no disk space for a torch install),
   partly deliberate, since deriving and gradient-checking the
   backward pass by hand (see `05`) is a stronger test of actually
   understanding what a conv layer's gradient is doing than importing
   one.

## Results (leave-one-well-out, 8 wells)

| Model | Mean accuracy | Std | Mean adjacent-facies accuracy |
|---|---|---|---|
| **DepthWindowCNN** | **0.506** | 0.075 | 0.806 |
| XGBoost | 0.454 | 0.147 | 0.825 |
| Random Forest | 0.450 | 0.159 | 0.825 |
| LDA (classical) | 0.447 | 0.122 | 0.837 |

Full per-well breakdown in `results/leave_one_well_out.csv` and
`notebooks/06_leave_one_well_out_comparison.ipynb`.

**The honest read of this, not the flattering one:**

- The CNN wins on mean accuracy and is the most *consistent*
  model well-to-well (lowest std) -- the strongest evidence in this
  project that depth-window context genuinely helps, not just a lucky
  split.
- It comes with a real trade-off: adjacent-facies accuracy is
  *lower* for the CNN than for all three tabular models. It's making
  fewer "safe," geologically-adjacent mistakes and more outright
  misses -- a plausible side effect of balanced class weighting
  pushing it to commit to minority-class predictions instead of
  defaulting to a neighboring majority class.
- LDA, Random Forest, and XGBoost land within a single point of
  each other on mean accuracy. A single-blind-well comparison
  (`03`/`04`) made LDA look meaningfully ahead of the tree models --
  that gap mostly disappears once averaged over all 8 wells, and
  turns out to have been specific to one well (SHANKLE), not a
  systematic pattern. That's the main reason `06` exists rather than
  reporting the single-well numbers as the final result.
- All four models fall apart on Recruit F9 (68 total samples, the
  shortest well by a wide margin) relative to the other 7 wells, but
  the CNN degrades far less (0.37 vs. 0.09-0.16 for the tabular
  models) -- consistent with, though not proof of, the idea that
  depth-window context gives it something to work with per-sample
  even when a well is too short for cross-well generalization to help
  much.

## What I'd do next with more time/compute

- Re-run `04`'s small XGBoost hyperparameter sweep (`max_depth=3,
  lr=0.03` slightly outperformed the defaults on the SHANKLE split)
  across the full leave-one-well-out harness to see if the edge holds
  or was split-specific.
- A second conv layer or a small residual connection in
  `DepthWindowCNN` -- kept to one layer here because with ~2800
  training rows per fold, a deeper net has more parameters than the
  data can support, but it's the first thing worth testing with a
  larger dataset.
- Investigate Recruit F9 and CROSS H CATTLE directly (log tracks,
  facies sequence) rather than treating "this well is hard for every
  model" as a black box.

## Reproducing this

```bash
pip install -r requirements.txt
python -m src.data_loader        # sanity-check the data loads
python -m src.run_lowo           # full leave-one-well-out CV, ~1 min, writes results/
jupyter notebook notebooks/      # or open individual notebooks directly
```

## Data source

`data/raw/well_logs_council_grove.csv` -- training wells with facies
labels, Council Grove Field, Kansas. `data/raw/blind_test_unlabeled.csv`
-- 2 additional wells (STUART, CRAWFORD) without facies labels, held
out by the original SEG contest for leaderboard scoring; not used
here since there's no ground truth to validate against, kept only for
reference. Both originate from the Kansas Geological Survey, compiled
for the SEG 2016 machine learning contest / Brendon Hall's tutorial.
=======
# WELL-LOGS-CLASSIFICATION-
Built a well-log facies classification pipeline (Kansas Geological Survey dataset, 8 wells, 9 facies classes) comparing LDA, Random Forest, XGBoost, and a from-scratch NumPy 1D-CNN
>>>>>>> 724ddd77d5dba2c6de8aa7f758dd4774fe8e8480
