"""
Facies confusions aren't all equally bad -- wackestone vs. packstone
is a judgment call even for a geologist looking at core; wackestone
vs. nonmarine sandstone is a real miss. adjacent_accuracy() gives
credit for landing on a geologically neighboring facies, which is
the metric the original SEG contest used alongside plain accuracy.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from src.data_loader import ADJACENT_FACIES


def adjacent_accuracy(y_true, y_pred):
    correct = 0
    for t, p in zip(y_true, y_pred):
        if p == t or p in ADJACENT_FACIES.get(t, []):
            correct += 1
    return correct / len(y_true)


def summarize(y_true, y_pred, model_name):
    acc = (y_true == y_pred).mean()
    adj = adjacent_accuracy(y_true, y_pred)
    print(f"{model_name:>22s}  |  accuracy {acc:.3f}  |  adjacent-facies accuracy {adj:.3f}")
    return {"model": model_name, "accuracy": acc, "adjacent_accuracy": adj}


def full_report(y_true, y_pred, labels, target_names):
    print(classification_report(y_true, y_pred, labels=labels,
                                 target_names=target_names, zero_division=0))
    return confusion_matrix(y_true, y_pred, labels=labels)
