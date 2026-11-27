"""
Train, cross-validate and predict.

  python train.py --model logreg                 # one model
  python train.py --model all                    # benchmark every model (CV table)
  python train.py --model logreg --min-active 0  # keep all signatures (no activity filter)

Design choices
  * Grouped CV: folds are made of whole compounds (cv_fold in meta_train.csv), so a compound
    is never in both the fitting and the validation part.
  * Activity filter: inactive signatures (e.g. very low doses) look like DMSO and only add
    noise. A signature is 'active' if >= --min-active genes have |z| > 2. Filtered signatures
    are dropped from TRAINING only; every test signature is still scored.
  * Compound-level aggregation: probabilities of all doses of a compound are averaged,
    weighted by each signature's activity (strong signatures count more).

Outputs
  results/cv_<model>.json                CV metrics (compound level)
  results/submission_<model>.csv         test predictions, one row per compound
  results/cv_benchmark.csv               with --model all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROC
from evaluate import compound_metrics
from models import MODELS, make_model

RES = Path("results")


def activity(X: pd.DataFrame) -> pd.Series:
    """Number of strongly changed genes (|z| > 2): a simple transcriptional activity score."""
    return (X.abs() > 2).sum(axis=1)


def load():
    Xtr = pd.read_csv(PROC / "X_train.csv.gz", index_col=0)
    ytr = pd.read_csv(PROC / "y_train.csv", index_col=0)["moa"].loc[Xtr.index]
    mtr = pd.read_csv(PROC / "meta_train.csv", index_col=0).loc[Xtr.index]
    Xte = pd.read_csv(PROC / "X_test.csv.gz", index_col=0)
    mte = pd.read_csv(PROC / "meta_test.csv", index_col=0).loc[Xte.index]
    return Xtr, ytr, mtr, Xte, mte


def aggregate(P: pd.DataFrame, compounds: pd.Series, weights: pd.Series) -> pd.DataFrame:
    w = weights.loc[P.index].astype(float) + 1.0              # +1: inactive signatures still count a little
    Pw = P.mul(w, axis=0)
    out = Pw.groupby(compounds.loc[P.index].values).sum()
    return out.div(w.groupby(compounds.loc[P.index].values).sum(), axis=0)


def fit_predict(name, Xfit, yfit, Xpred, min_active):
    act = activity(Xfit)
    keep = act >= min_active
    if keep.sum() < 10 or yfit[keep].nunique() < yfit.nunique():
        keep[:] = True                                       # never lose a class to the filter
    m = make_model(name)
    try:
        m.fit(Xfit[keep], yfit[keep])
    except (ValueError, TypeError):                          # e.g. MLP early stopping on tiny classes
        if name != "mlp":
            raise
        m.est.set_params(mlpclassifier__early_stopping=False)
        m.fit(Xfit[keep], yfit[keep])
    return m, m.predict_proba(Xpred)


def cross_validate(name, Xtr, ytr, mtr, min_active):
    parts = []
    for f in sorted(mtr["cv_fold"].unique()):
        va = mtr["cv_fold"] == f
        _, P = fit_predict(name, Xtr[~va], ytr[~va], Xtr[va], min_active)
        parts.append(P)
    P = pd.concat(parts).reindex(columns=sorted(ytr.unique())).fillna(0.0)
    scores = aggregate(P, mtr["compound"], activity(Xtr))
    truth = pd.Series(ytr.values, index=mtr["compound"].values).groupby(level=0).first()
    return compound_metrics(scores, truth)


def run(name, min_active):
    Xtr, ytr, mtr, Xte, mte = load()
    RES.mkdir(exist_ok=True)
    cv = cross_validate(name, Xtr, ytr, mtr, min_active)
    print(f"[{name}] grouped CV (compound level): {cv}")
    (RES / f"cv_{name}.json").write_text(json.dumps(cv, indent=2))

    model, P = fit_predict(name, Xtr, ytr, Xte, min_active)
    scores = aggregate(P, mte["compound"], activity(Xte))
    scores.index.name = "compound"
    scores.reset_index().to_csv(RES / f"submission_{name}.csv", index=False)
    print(f"[{name}] wrote results/submission_{name}.csv ({len(scores)} test compounds)")
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logreg", choices=MODELS + ["all"])
    ap.add_argument("--min-active", type=int, default=20,
                    help="min genes with |z|>2 for a training signature (0 = keep all)")
    a = ap.parse_args()
    names = MODELS if a.model == "all" else [a.model]
    rows = [{"model": n, **run(n, a.min_active)} for n in names]
    if len(rows) > 1:
        tab = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
        tab.to_csv(RES / "cv_benchmark.csv", index=False)
        print("\n" + tab.to_string(index=False))


if __name__ == "__main__":
    main()
