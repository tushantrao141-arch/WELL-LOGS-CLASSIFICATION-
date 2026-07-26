"""
Feature prep.

Two things matter more here than in a generic tabular problem:

1. Log calibration drifts between wells (different tools, different
   runs, sometimes different service companies). Z-scoring each curve
   *within* a well before pooling wells together removes a lot of
   that systematic offset. Do this on train only and apply the same
   per-well stats to the blind well -- in a real deployment you would
   not have facies labels for the blind well to lean on, but you
   would have its own logs, so per-well normalization is still fair.

2. Facies is not a point property -- a given depth's rock type is
   correlated with what is above and below it (beds have thickness).
   A plain row-by-row classifier throws that away. Rolling-window
   stats per curve give nearby-depth context without needing a full
   sequence model.
"""

import numpy as np
import pandas as pd
from src.data_loader import LOG_CURVES


def normalize_by_well(df, curves=LOG_CURVES[:-2]):
    """Z-score GR, ILD_log10, DeltaPHI, PHIND, PE within each well.
    NM_M is a binary marine/nonmarine flag and RELPOS is already a
    0-1 relative position within a depositional cycle -- neither
    needs normalizing."""
    out = df.copy()
    for well, group in df.groupby("Well Name", observed=True):
        idx = group.index
        for c in curves:
            mu, sd = group[c].mean(), group[c].std()
            sd = sd if sd > 1e-6 else 1.0
            out.loc[idx, c] = (group[c] - mu) / sd
    return out


def add_window_features(df, curves=("GR", "ILD_log10", "PHIND"), window=5):
    """Rolling mean/std per curve, computed within each well so a
    window never bleeds across a well boundary. window is in samples
    (logs here are sampled every 0.5 ft, so window=5 is a 2.5 ft
    smoothing window -- roughly one bed)."""
    out = df.copy()
    for well, group in df.groupby("Well Name", observed=True):
        idx = group.index
        for c in curves:
            roll = group[c].rolling(window, center=True, min_periods=1)
            out.loc[idx, f"{c}_rmean"] = roll.mean().values
            out.loc[idx, f"{c}_rstd"] = roll.std().fillna(0).values
        # gradient gives a crude sense of "am I near a bed boundary"
        for c in curves:
            out.loc[idx, f"{c}_grad"] = np.gradient(group[c].values)
    return out


def build_feature_matrix(df, window=5):
    df = normalize_by_well(df)
    df = add_window_features(df, window=window)
    feature_cols = [c for c in df.columns if c not in
                    ("Facies", "Formation", "Well Name", "Depth")]
    X = df[feature_cols].values.astype(np.float64)
    y = df["Facies"].values.astype(np.int64) if "Facies" in df.columns else None
    return X, y, feature_cols


def sliding_sequences(df, curves=LOG_CURVES, window=9, normalize=True):
    """Build (n_samples, window, n_curves) tensors per well for the
    from-scratch CNN -- each sample is a depth window centered on the
    labeled depth, padded with edge-replication at well tops/bottoms
    so we don't lose the first/last few samples of each well.

    normalize=True runs the same per-well z-scoring used for the
    tabular models first. Turn it off only if you're passing in a
    frame that's already been through normalize_by_well -- raw GR
    values run into the hundreds and will blow up training."""
    if normalize:
        df = normalize_by_well(df, curves=[c for c in curves if c not in ("NM_M", "RELPOS")])
    half = window // 2
    seqs, labels = [], []
    for well, group in df.groupby("Well Name", observed=True):
        group = group.sort_values("Depth")
        arr = group[curves].values.astype(np.float64)
        padded = np.pad(arr, ((half, half), (0, 0)), mode="edge")
        for i in range(len(arr)):
            seqs.append(padded[i:i + window])
            if "Facies" in group.columns:
                labels.append(group["Facies"].values[i])
    X = np.stack(seqs)
    y = np.array(labels, dtype=np.int64) if labels else None
    return X, y
