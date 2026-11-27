"""
Template. Fill in the three TODOs and you have a submission.

  python solution_template.py

Structure of the task, so nothing is mysterious:
  input  : a fingerprint = how 978 genes responded to one drug at one dose
  output : for each unknown drug, a score per mechanism class
  rule   : a drug's doses are all on the same side of any split, never both

The check at the bottom tells you whether your idea is actually better than the
baseline, using only training data. Do not tune against the leaderboard.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA = "/data/cureatime"          # change to "data/processed" if you run it locally

X_train = pd.read_csv(f"{DATA}/X_train.csv.gz", index_col=0)
y_train = pd.read_csv(f"{DATA}/y_train.csv", index_col=0)["moa"]
meta_train = pd.read_csv(f"{DATA}/meta_train.csv", index_col=0)
X_test = pd.read_csv(f"{DATA}/X_test.csv.gz", index_col=0)
meta_test = pd.read_csv(f"{DATA}/meta_test.csv", index_col=0)
CLASSES = sorted(y_train.unique())


# ---------------------------------------------------------------- TODO 1
def preprocess(X):
    """Clean up the fingerprints however you like. Return the same shape.

    Ideas: drop genes that barely move; scale each row; keep only pathway genes.
    Leave it as-is to start.
    """
    return X


# ---------------------------------------------------------------- TODO 2
def fit_and_score(X_fit, y_fit, X_pred):
    """Learn from (X_fit, y_fit); return a DataFrame of scores: rows = X_pred, cols = CLASSES.

    The version below is the baseline: similarity to each class's average fingerprint.
    Replace it with anything, e.g.

        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")
        m.fit(X_fit, y_fit)
        return pd.DataFrame(m.predict_proba(X_pred), index=X_pred.index, columns=m.classes_)
    """
    profiles = X_fit.groupby(y_fit).mean()
    unit = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    sim = unit(X_pred[profiles.columns].values) @ unit(profiles.values).T
    return pd.DataFrame(sim, index=X_pred.index, columns=profiles.index)


# ---------------------------------------------------------------- TODO 3
def combine_doses(scores, meta):
    """One row per drug from its several doses. Plain average to start.

    Ideas: ignore doses where almost no gene moved; weight by how much moved.
    """
    return scores.groupby(meta.loc[scores.index, "compound"]).mean()


# ---------------------------------------------------------------- check
def self_check():
    """Honest score on training data only, using the supplied folds."""
    Xp = preprocess(X_train)
    per_fold = []
    for fold in sorted(meta_train["cv_fold"].unique()):
        out = meta_train["cv_fold"] == fold
        per_fold.append(fit_and_score(Xp[~out], y_train[~out], Xp[out]))
    scores = pd.concat(per_fold).reindex(columns=CLASSES).fillna(0.0)

    pred = combine_doses(scores, meta_train).idxmax(axis=1)
    truth = y_train.groupby(meta_train["compound"]).first()
    pred = pred.reindex(truth.index)
    print(f"macro-F1 on held-out drugs: {f1_score(truth, pred, average='macro'):.3f}")
    print("\nwhere it goes wrong:")
    print(pd.crosstab(truth, pred, rownames=["true"], colnames=["predicted"]))


def make_submission():
    Xp_train, Xp_test = preprocess(X_train), preprocess(X_test)
    scores = fit_and_score(Xp_train, y_train, Xp_test).reindex(columns=CLASSES).fillna(0.0)
    sub = combine_doses(scores, meta_test)
    sub.index.name = "compound"
    sub.reset_index().to_csv("submission.csv", index=False)
    print(f"\nwrote submission.csv for {len(sub)} drugs")


if __name__ == "__main__":
    self_check()
    make_submission()
