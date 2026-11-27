"""
Interpretation helpers (useful for biology participants).

  python interpret.py

Outputs (results/)
  class_signatures.csv      top-15 up and down genes per mechanism (mean z over training compounds)
  test_neighbours.csv       for every test compound: the 5 most similar training compounds
                            and their mechanisms (cosine similarity of compound signatures)
"""
import numpy as np
import pandas as pd

from config import PROC
from train import RES, activity, load


def compound_profiles(X, meta, min_active=20):
    """One profile per compound: mean of its active signatures (all if none are active)."""
    act = activity(X)
    rows = {}
    for cpd, idx in meta.groupby("compound").groups.items():
        a = act.loc[idx]
        use = a[a >= min_active].index if (a >= min_active).any() else idx
        rows[cpd] = X.loc[use].mean()
    return pd.DataFrame(rows).T


def main():
    Xtr, ytr, mtr, Xte, mte = load()
    RES.mkdir(exist_ok=True)
    Ptr = compound_profiles(Xtr, mtr)
    ycpd = pd.Series(ytr.values, index=mtr["compound"].values).groupby(level=0).first().loc[Ptr.index]

    rows = []
    for cls, cpds in ycpd.groupby(ycpd):
        mean = Ptr.loc[cpds.index].mean()
        up, down = mean.nlargest(15), mean.nsmallest(15)
        rows.append({"mechanism": cls, "n_compounds": len(cpds),
                     "top_up": ", ".join(f"{g}({v:.1f})" for g, v in up.items()),
                     "top_down": ", ".join(f"{g}({v:.1f})" for g, v in down.items())})
    pd.DataFrame(rows).to_csv(RES / "class_signatures.csv", index=False)

    Pte = compound_profiles(Xte, mte)
    U = lambda M: M.values / (np.linalg.norm(M.values, axis=1, keepdims=True) + 1e-9)
    S = U(Pte) @ U(Ptr).T
    nb = []
    for i, cpd in enumerate(Pte.index):
        top = np.argsort(-S[i])[:5]
        nb.append({"test_compound": cpd,
                   **{f"nn{k + 1}": f"{Ptr.index[j]} [{ycpd.iloc[j]}] {S[i, j]:.2f}" for k, j in enumerate(top)}})
    pd.DataFrame(nb).to_csv(RES / "test_neighbours.csv", index=False)
    print(pd.DataFrame(rows)[["mechanism", "n_compounds", "top_down"]].head(20).to_string(index=False))
    print("-> results/class_signatures.csv, results/test_neighbours.csv")


if __name__ == "__main__":
    main()
