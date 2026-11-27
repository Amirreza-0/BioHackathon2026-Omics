"""
Preprocess L1000 signatures into the task dataset and make compound-level splits.

  python preprocess.py

Steps
  1. read level-5 consensus signatures (one per compound x dose, 978 genes, robust z-scores)
  2. keep compound treatments only (Broad IDs), parse dose in uM
  3. attach MoA labels; keep single-mechanism compounds in TARGET_CLASSES
  4. drop classes with < MIN_COMPOUNDS_PER_CLASS compounds
  5. clip extreme z-scores to [-10, 10]
  6. split by COMPOUND (stratified by class): train / hidden test; CV folds inside train
  7. save everything else as unlabeled data

Output (data/processed/)
  X_train.csv.gz, y_train.csv            features + labels (rows = signatures)
  X_test.csv.gz                          features of the hidden test set
  X_unlabeled.csv.gz                     all other compounds, no labels
  meta_train.csv, meta_test.csv          sample_id, compound, dose_um, cv_fold (train)
  classes.txt
  organizers_only/y_test.csv             hidden labels
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config import (CV_FOLDS, L1000_LEVEL5, L1000_LEVEL5_META, MIN_COMPOUNDS_PER_CLASS,
                    MIN_TEST_PER_CLASS, PROC, RAW, SEED, SINGLE_MOA_ONLY, TARGET_CLASSES,
                    TEST_FRAC)


def compound_id(x: pd.Series) -> pd.Series:
    """'BRD-K12345678-001-01-5' -> 'BRD-K12345678' (same molecule across batches/salts)."""
    return x.astype(str).str[:13]


def parse_dose(s) -> float:
    m = re.search(r"([\d.]+)\s*(uM|um|µM|nM)?", str(s))
    if not m:
        return np.nan
    v = float(m.group(1))
    return v / 1000 if (m.group(2) or "").lower() == "nm" else v


def read_gctx(path) -> pd.DataFrame:
    """CMap GCTX (HDF5) -> DataFrame signatures x genes (gene symbols if available)."""
    import h5py

    dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]
    with h5py.File(path, "r") as f:
        mat = f["0/DATA/0/matrix"][()]                     # stored as columns x rows
        cols = dec(f["0/META/COL/id"][()])
        rows = dec(f["0/META/ROW/id"][()])
        row_meta = f["0/META/ROW"]
        sym = dec(row_meta["pr_gene_symbol"][()]) if "pr_gene_symbol" in row_meta else rows
    return pd.DataFrame(mat.astype("float32"), index=cols, columns=sym)


def load_signatures() -> pd.DataFrame:
    l1 = RAW / "l1000"
    X = read_gctx(next(l1.rglob(L1000_LEVEL5)))
    meta = pd.read_csv(next(l1.rglob(L1000_LEVEL5_META)), sep="\t").set_index("sig_id")
    meta = meta.loc[meta.index.intersection(X.index)]
    meta = meta[meta["pert_id"].astype(str).str.startswith("BRD-")]
    df = pd.DataFrame({"compound": compound_id(meta["pert_id"]).values,
                       "dose_um": meta["pert_idose"].map(parse_dose).values}, index=meta.index)
    df = df.join(X.loc[meta.index].clip(-10, 10))
    df.index.name = "sample_id"
    print(f"{len(df)} compound signatures, {df.compound.nunique()} compounds, {X.shape[1]} genes")
    return df


def load_labels(compounds_with_data) -> pd.Series:
    moa = pd.read_csv(RAW / "moa_labels.tsv", sep="\t")
    moa["compound"] = compound_id(moa["broad_id"])
    moa = moa.dropna(subset=["moa"]).drop_duplicates("compound")
    moa = moa[moa["compound"].isin(compounds_with_data)]
    all_labels = moa.set_index("compound")["moa"]
    if SINGLE_MOA_ONLY:
        moa = moa[~moa["moa"].str.contains(r"\|")]
    moa = moa[moa["moa"].isin(TARGET_CLASSES)]
    vc = moa["moa"].value_counts()
    keep = vc[vc >= MIN_COMPOUNDS_PER_CLASS]
    print("Classes kept (compounds):\n" + keep.to_string())
    dropped = sorted(set(TARGET_CLASSES) - set(keep.index))
    if dropped:
        print("Dropped (too few compounds):", dropped)
    return moa[moa["moa"].isin(keep.index)].set_index("compound")["moa"], all_labels


def split_compounds(labels: pd.Series) -> pd.DataFrame:
    """Per class: TEST_FRAC of compounds -> test; the rest -> train with CV folds
    assigned round-robin within class (stratified, grouped by compound)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for cls, cpds in labels.groupby(labels):
        c = rng.permutation(cpds.index.values)
        n_test = max(MIN_TEST_PER_CLASS, int(round(len(c) * TEST_FRAC)))
        for i, cpd in enumerate(c):
            if i < n_test:
                rows.append((cpd, cls, "test", -1))
            else:
                rows.append((cpd, cls, "train", (i - n_test) % CV_FOLDS))
    return pd.DataFrame(rows, columns=["compound", "moa", "split", "cv_fold"]).set_index("compound")


def main():
    df = load_signatures()
    labels, all_labels = load_labels(set(df.compound))
    sp = split_compounds(labels)

    out, org = PROC, PROC / "organizers_only"
    org.mkdir(parents=True, exist_ok=True)
    genes = [c for c in df.columns if c not in ("compound", "dose_um")]

    lab = df[df.compound.isin(sp.index)].copy()
    lab = lab.join(sp, on="compound")
    tr, te = lab[lab.split == "train"], lab[lab.split == "test"]

    tr[genes].to_csv(out / "X_train.csv.gz")
    tr[["moa"]].to_csv(out / "y_train.csv")
    tr[["compound", "dose_um", "cv_fold"]].to_csv(out / "meta_train.csv")
    te[genes].to_csv(out / "X_test.csv.gz")
    te[["compound", "dose_um"]].to_csv(out / "meta_test.csv")
    te[["compound", "moa"]].to_csv(org / "y_test.csv")

    # Unlabeled = every compound not in the benchmark (never a test compound)
    un = df[~df.compound.isin(sp.index)]
    un[["compound", "dose_um"] + genes].to_csv(out / "X_unlabeled.csv.gz")
    # Organizers can use full labels of unlabeled compounds for analysis, teams cannot
    all_labels.rename("moa_any").to_csv(org / "all_moa_labels.csv")

    (out / "classes.txt").write_text("\n".join(sorted(labels.unique())) + "\n")
    n = sp.groupby(["moa", "split"]).size().unstack(fill_value=0)
    print("\nCompounds per class and split:\n" + n.to_string())
    print(f"\ntrain {len(tr)} signatures / {tr.compound.nunique()} compounds | "
          f"test {len(te)} / {te.compound.nunique()} | unlabeled {len(un)} / {un.compound.nunique()}")


if __name__ == "__main__":
    main()
