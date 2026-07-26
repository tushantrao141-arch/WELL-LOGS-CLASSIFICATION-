"""
Leave-one-well-out cross validation, all four models, all eight wells.
One arbitrary blind well (as used in the individual notebooks) is a
nice narrative but eight is what actually tells you whether a result
is real or just a lucky split. Run as:  python -m src.run_lowo
"""
import json
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import load_raw, well_list, split_blind_well
from src.features import build_feature_matrix, sliding_sequences
from src.models import get_random_forest, get_xgboost, xgb_sample_weights, DepthWindowCNN
from src.evaluate import adjacent_accuracy


def run():
    df = load_raw()
    wells = well_list(df)
    classes = sorted(df["Facies"].unique())
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    rows = []
    for well in wells:
        train, blind = split_blind_well(df, well)
        if len(blind) < 20:
            continue  # guards against a degenerate blind split; none of the 8 wells actually hit this

        Xtr, ytr, _ = build_feature_matrix(train)
        Xbl, ybl, _ = build_feature_matrix(blind)

        lda = LinearDiscriminantAnalysis()
        lda.fit(Xtr, ytr)
        rows.append(dict(well=well, model="LDA (classical)",
                          accuracy=(lda.predict(Xbl) == ybl).mean(),
                          adj_accuracy=adjacent_accuracy(ybl, lda.predict(Xbl))))

        rf = get_random_forest()
        rf.fit(Xtr, ytr)
        rows.append(dict(well=well, model="Random Forest",
                          accuracy=(rf.predict(Xbl) == ybl).mean(),
                          adj_accuracy=adjacent_accuracy(ybl, rf.predict(Xbl))))

        xgb = get_xgboost()
        sw = xgb_sample_weights(ytr)
        xgb.fit(Xtr, ytr - 1, sample_weight=sw)
        xpred = xgb.predict(Xbl) + 1
        rows.append(dict(well=well, model="XGBoost",
                          accuracy=(xpred == ybl).mean(),
                          adj_accuracy=adjacent_accuracy(ybl, xpred)))

        Xtr_seq, ytr_seq = sliding_sequences(train, window=9)
        Xbl_seq, ybl_seq = sliding_sequences(blind, window=9)
        ytr_idx = np.array([cls_to_idx[c] for c in ytr_seq])
        ybl_idx = np.array([cls_to_idx[c] for c in ybl_seq])
        sw_cnn = compute_sample_weight("balanced", ytr_idx)
        cnn = DepthWindowCNN(window=9, n_curves=Xtr_seq.shape[2], n_classes=len(classes))
        cnn.fit(Xtr_seq, ytr_idx, sw_cnn, X_val=Xbl_seq, y_val_idx=ybl_idx,
                epochs=40, lr=0.008, lr_decay=0.98, seed=hash(well) % 1000)
        cpred_idx = cnn.predict(Xbl_seq)
        idx_to_cls = {i: c for c, i in cls_to_idx.items()}
        cpred = np.array([idx_to_cls[i] for i in cpred_idx])
        rows.append(dict(well=well, model="DepthWindowCNN",
                          accuracy=(cpred == ybl_seq).mean(),
                          adj_accuracy=adjacent_accuracy(ybl_seq, cpred)))

        print(f"done: {well}")

    out = pd.DataFrame(rows)
    out.to_csv("results/leave_one_well_out.csv", index=False)

    summary = out.groupby("model")[["accuracy", "adj_accuracy"]].agg(["mean", "std"])
    print(summary)
    summary.to_csv("results/lowo_summary.csv")
    return out, summary


if __name__ == "__main__":
    run()
