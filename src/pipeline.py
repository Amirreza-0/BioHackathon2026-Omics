"""
cureATime / DFKI omics challenge - mechanism-of-action prediction from L1000.

Reproduces the final submission end to end:

    python src/pipeline.py --data /path/to/omics --out submission/submission_final_v2.csv

Final model = 50/50 blend of
  (a) cosine centroid on calibrated compound profiles, with soft pathway-superclass
      labels plus mechanistic evidence terms for PARP and RAF, and
  (b) LightGBM trained at signature level,
followed by a prior-corrected decision rule (score_c / prior_c ** 0.5).

Cross-validated macro-F1 0.728 +/- 0.014 (nested, 10 repeated grouped 4-fold splits);
0.715 on ten further seeds held out from that selection.
"""
import argparse, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

CTRL = "BRD-K60230970"          # 300-replicate plate control; excluded from training
HI_DOSES = [3.33, 10.0]         # doses averaged into a compound profile
LOW_DOSES = [0.04, 0.12, 0.37, 1.11]
TOP_DOSE = 10.0
ALPHA = 0.5                     # prior-correction exponent (selected by nested CV)
BLEND_W = 0.5                   # weight on the LightGBM arm

# ERK output module: the six landmark genes MEK inhibition suppresses most.
# DUSP6, DUSP4, IER3, ETV1, FOSL1, SPRED2
ERK_PROBES = ['208891_at', '204014_at', '201631_s_at', '206501_x_at', '204420_at', '212458_at']

# p53 effector module, used by the PARP evidence term.
# CDKN1A, DDB2, FAS, SESN1, BAX, GADD45A
P53_PROBES = ['202284_s_at', '203409_at', '204781_s_at', '218346_s_at', '208478_s_at', '203725_at']

SUPERCLASS = {
    "PI3K inhibitor": "PI3Kaxis", "AKT inhibitor": "PI3Kaxis", "mTOR inhibitor": "PI3Kaxis",
    "EGFR inhibitor": "MAPK", "RAF inhibitor": "MAPK", "MEK inhibitor": "MAPK",
    "topoisomerase inhibitor": "DNAdamage", "PARP inhibitor": "DNAdamage",
    "tubulin polymerization inhibitor": "mitotic", "PLK inhibitor": "mitotic",
    "CDK inhibitor": "cellcycle", "HDAC inhibitor": "chromatin",
    "proteasome inhibitor": "proteostasis", "HSP inhibitor": "proteostasis",
    "JAK inhibitor": "cytokine",
}

LGB_PARAMS = dict(n_estimators=456, num_leaves=5, max_depth=3, learning_rate=0.085,
                  min_child_samples=5, subsample=0.82, subsample_freq=1,
                  colsample_bytree=0.24, reg_lambda=7.69, class_weight="balanced",
                  objective="multiclass", verbose=-1, n_jobs=4)


# ---------------------------------------------------------------- data / calibration
def load(data_dir):
    d = lambda f: f"{data_dir}/{f}"
    X_tr = pd.read_csv(d("X_train.csv.gz"))
    X_te = pd.read_csv(d("X_test.csv.gz"))
    X_un = pd.read_csv(d("X_unlabeled.csv.gz"))
    y_tr = pd.read_csv(d("y_train.csv"))
    m_tr = pd.read_csv(d("meta_train.csv"))
    m_te = pd.read_csv(d("meta_test.csv"))
    classes = [l.strip() for l in open(d("classes.txt")) if l.strip()]
    genes = [c for c in X_tr.columns if c != "sample_id"]
    return X_tr, X_te, X_un, y_tr, m_tr, m_te, classes, genes


def compound_profiles(X, meta, genes):
    """One profile per compound: mean of its 3.33 and 10 uM signatures."""
    df = meta[["sample_id", "compound", "dose_um"]].merge(X, on="sample_id")
    return df[df.dose_um.isin(HI_DOSES)].groupby("compound")[genes].mean()


def fit_calibration(X_un, genes):
    """Per-gene median and MAD across the unlabelled library (the 'inert' reference)."""
    un = X_un[X_un.compound != CTRL]
    prof = compound_profiles(un[["sample_id"] + genes], un[["sample_id", "compound", "dose_um"]], genes)
    med = prof.median(0).values
    mad = (prof - prof.median(0)).abs().median(0).values * 1.4826
    return med, np.where(mad < 1e-6, 1.0, mad)


# ---------------------------------------------------------------- model components
def soft_targets(labels, classes, eps=0.15):
    """Put eps of each label's mass on its pathway-superclass siblings."""
    idx = {c: i for i, c in enumerate(classes)}
    T = np.zeros((len(labels), len(classes)))
    for i, c in enumerate(labels):
        sibs = [d for d in classes if SUPERCLASS[d] == SUPERCLASS[c] and d != c]
        T[i, idx[c]] = 1 - eps if sibs else 1.0
        for s in sibs:
            T[i, idx[s]] = eps / len(sibs)
    return T


def unit(Z):
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.where(n < 1e-9, 1, n)


def centroid_proba(Z_tr, y_tr, Z_q, classes, eps=0.15, temp=0.1):
    cen = unit(soft_targets(y_tr, classes, eps).T @ unit(Z_tr))
    S = unit(Z_q) @ cen.T
    E = np.exp((S - S.max(1, keepdims=True)) / temp)
    return E / E.sum(1, keepdims=True)


def paradox_index(curves):
    """Up-then-down ERK crossover: needs a positive low-dose peak AND a fall at top dose."""
    mx = curves[LOW_DOSES].max(1)
    return np.minimum(mx, mx - curves[TOP_DOSE])


def dose_curves(X, meta, probes, genes, med, mad):
    """Per-compound module score at each dose."""
    df = meta[["sample_id", "compound", "dose_um"]].merge(X, on="sample_id")
    Z = (df[genes].values - med) / mad
    gi = [genes.index(p) for p in probes]
    df = df.assign(score=Z[:, gi].mean(1))
    piv = df.pivot_table(index="compound", columns="dose_um", values="score")
    return piv.T.interpolate(limit_direction="both").T       # one compound has 5 doses


def evidence_term(X_fit, y_bin, X_q):
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_fit, y_bin)
    return lr.predict_proba(X_q)[:, 1]


# ---------------------------------------------------------------- main
def main(data_dir, out_path):
    X_tr, X_te, X_un, y_tr, m_tr, m_te, classes, genes = load(data_dir)
    ci = {c: i for i, c in enumerate(classes)}
    med, mad = fit_calibration(X_un, genes)

    meta = m_tr.merge(y_tr, on="sample_id")
    meta = meta[meta.compound != CTRL].reset_index(drop=True)   # drop the plate control

    P_tr = compound_profiles(X_tr, meta, genes)
    P_te = compound_profiles(X_te, m_te, genes)
    Z_tr, Z_te = (P_tr.values - med) / mad, (P_te.values - med) / mad
    comps, test_comps = P_tr.index.to_numpy(), P_te.index.to_numpy()
    y = meta.groupby("compound").moa.first().reindex(comps).to_numpy()

    sig = meta[["sample_id", "compound", "dose_um", "moa"]].merge(X_tr, on="sample_id")
    Z_sig = (sig[genes].values - med) / mad
    te_sig = m_te[["sample_id", "compound", "dose_um"]].merge(X_te, on="sample_id")
    Z_te_sig = (te_sig[genes].values - med) / mad

    # --- mechanistic features -------------------------------------------------
    erk_tr = dose_curves(X_tr, meta, ERK_PROBES, genes, med, mad).reindex(comps)
    erk_te = dose_curves(X_te, m_te, ERK_PROBES, genes, med, mad).reindex(test_comps)
    raf_tr = paradox_index(erk_tr).values[:, None]
    raf_te = paradox_index(erk_te).values[:, None]

    p53_i = [genes.index(p) for p in P53_PROBES]
    amp = lambda Z, comp, idx: np.log10(pd.Series((np.abs(Z) > 2).sum(1), index=comp)
                                        .groupby(level=0).max().reindex(idx).values + 1)
    parp_tr = np.c_[Z_tr[:, p53_i].mean(1), amp(Z_sig, sig.compound.values, comps)]
    parp_te = np.c_[Z_te[:, p53_i].mean(1), amp(Z_te_sig, te_sig.compound.values, test_comps)]

    # --- arm 1: centroid ------------------------------------------------------
    Pc = centroid_proba(Z_tr, y, Z_te, classes).copy()
    Pc[:, ci["PARP inhibitor"]] = (0.9 * Pc[:, ci["PARP inhibitor"]]
        + 0.1 * evidence_term(parp_tr, (y == "PARP inhibitor").astype(int), parp_te))
    Pc[:, ci["RAF inhibitor"]] = (0.7 * Pc[:, ci["RAF inhibitor"]]
        + 0.3 * evidence_term(raf_tr, (y == "RAF inhibitor").astype(int), raf_te))
    Pc /= Pc.sum(1, keepdims=True)

    # --- arm 2: signature-level LightGBM --------------------------------------
    mdl = lgb.LGBMClassifier(num_class=len(classes), **LGB_PARAMS)
    mdl.fit(Z_sig, np.array([ci[c] for c in sig.moa]))
    pg = pd.DataFrame(mdl.predict_proba(Z_te_sig))
    pg["compound"] = te_sig.compound.values
    Pg = pg.groupby("compound").mean().reindex(test_comps).values

    # --- blend + prior-corrected decision rule --------------------------------
    P = BLEND_W * Pg + (1 - BLEND_W) * Pc
    prior = np.array([(y == c).mean() for c in classes])
    P = P / prior ** ALPHA
    P /= P.sum(1, keepdims=True)

    sub = pd.DataFrame(P, columns=classes)
    sub.insert(0, "compound", test_comps)
    sub.to_csv(out_path, index=False)

    assert len(sub) == len(test_comps)
    assert list(sub.columns) == ["compound"] + classes
    assert np.allclose(sub[classes].sum(1), 1.0)
    print(f"wrote {out_path}: {len(sub)} rows, {len(classes)} classes, rows sum to 1.0")
    return sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory holding X_train.csv.gz etc.")
    ap.add_argument("--out", default="submission_final_v2.csv")
    main(ap.parse_args().data, ap.parse_args().out)
