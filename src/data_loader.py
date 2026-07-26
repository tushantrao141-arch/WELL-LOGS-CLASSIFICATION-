"""
Loading and basic bookkeeping for the Council Grove field well log set.

Nine facies classes, eight wells. The dataset is small enough that a
single held-out well is a reasonable proxy for a "blind well" test,
which is the standard way this kind of model actually gets validated
in practice (you never have log-response labels for the well you're
about to drill through).
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "well_logs_council_grove.csv"

FACIES_LABELS = {
    1: "Nonmarine sandstone",
    2: "Nonmarine coarse siltstone",
    3: "Nonmarine fine siltstone",
    4: "Marine siltstone/shale",
    5: "Mudstone",
    6: "Wackestone",
    7: "Dolomite",
    8: "Packstone/grainstone",
    9: "Phylloid-algal bafflestone",
}

# Facies that are geologically adjacent to one another. Confusing a
# wackestone for a packstone is a much smaller mistake than confusing
# it for a nonmarine sandstone, so this gets used later on for a more
# forgiving accuracy metric than raw top-1.
ADJACENT_FACIES = {
    1: [2], 2: [1, 3], 3: [2], 4: [3, 5], 5: [4, 6],
    6: [5, 7, 8], 7: [6, 8], 8: [6, 7, 9], 9: [7, 8],
}

LOG_CURVES = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]


def load_raw():
    df = pd.read_csv(RAW_PATH)
    df["Well Name"] = df["Well Name"].astype("category")
    return df


def well_list(df):
    return sorted(df["Well Name"].unique().tolist())


def split_blind_well(df, well_name):
    """Pull one well out entirely -- this is the train/test split that
    actually means something for this problem, as opposed to a random
    row split which would let the model cheat off neighbouring depths
    of the same well."""
    train = df[df["Well Name"] != well_name].reset_index(drop=True)
    blind = df[df["Well Name"] == well_name].reset_index(drop=True)
    return train, blind


if __name__ == "__main__":
    df = load_raw()
    print(f"{len(df)} rows, {df['Well Name'].nunique()} wells")
    print(df.groupby("Well Name", observed=True).size())
