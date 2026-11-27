"""
Score predictions at the COMPOUND level.

  python evaluate.py submission.csv data/processed/organizers_only/y_test.csv

submission.csv: one row per test compound: column `compound` + one column per class
(scores or probabilities). A signature-level file (column sample_id + compound) also
works: scores are averaged per compound first.

Metrics
  macro_f1           primary: F1 per class, averaged (every mechanism counts equally)
  balanced_accuracy  mean recall per class
  top1 / top3        is the true class the top / among the top-3 predictions
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


def to_compound_level(sub: pd.DataFrame) -> pd.DataFrame:
    if "compound" not in sub.columns:
        raise ValueError("submission needs a 'compound' column")
    num = sub.drop(columns=[c for c in ("sample_id", "compound", "dose_um") if c in sub.columns])
    return num.groupby(sub["compound"].values).mean()


def compound_metrics(scores: pd.DataFrame, truth: pd.Series) -> dict:
    """scores: compounds x classes; truth: compound -> class. Missing compounds count as wrong."""
    classes = list(scores.columns)
    scores = scores.reindex(truth.index).fillna(0.0)
    S = scores.values
    pred = np.array(classes, dtype=object)[S.argmax(axis=1)]
    pred[(S == 0).all(axis=1)] = "__missing__"
    y = truth.values
    order = np.argsort(-S, axis=1)
    idx = np.array([classes.index(t) if t in classes else -1 for t in y])
    return {
        "n_compounds": int(len(y)),
        "macro_f1": round(float(f1_score(y, pred, average="macro", labels=sorted(set(y)), zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "top1": round(float(np.mean(order[:, 0] == idx)), 4),
        "top3": round(float(np.mean([i in o[:3] for i, o in zip(idx, order)])), 4),
    }


def per_class_report(scores: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:
    scores = scores.reindex(truth.index).fillna(0.0)
    pred = scores.idxmax(axis=1)
    rows = []
    for c in sorted(truth.unique()):
        t, p = truth == c, pred == c
        rec = (t & p).sum() / t.sum()
        prec = (t & p).sum() / p.sum() if p.sum() else 0.0
        rows.append({"class": c, "n": int(t.sum()), "recall": round(rec, 3), "precision": round(prec, 3)})
    return pd.DataFrame(rows)


def confusion(scores: pd.DataFrame, truth: pd.Series) -> pd.DataFrame:
    scores = scores.reindex(truth.index).fillna(0.0)
    labels = sorted(truth.unique())
    pred = scores.idxmax(axis=1)
    return pd.DataFrame(confusion_matrix(truth, pred, labels=labels), index=labels, columns=labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("labels")
    ap.add_argument("--report", action="store_true", help="also print per-class table and confusion matrix")
    a = ap.parse_args()

    sub = pd.read_csv(a.submission)
    scores = to_compound_level(sub)
    truth = pd.read_csv(a.labels).drop_duplicates("compound").set_index("compound")["moa"]
    print(json.dumps(compound_metrics(scores, truth), indent=2))
    if a.report:
        print(per_class_report(scores, truth).to_string(index=False))
        print(confusion(scores, truth).to_string())


if __name__ == "__main__":
    main()
