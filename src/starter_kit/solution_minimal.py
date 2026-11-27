"""
The simplest complete answer. Run it, get a valid submission.

  python solution_minimal.py

The whole idea in one line: average the fingerprints of each mechanism class, then give
each unknown drug the class whose average its own fingerprint most resembles.

This is a real method (it is how the Connectivity Map works), it is one of the two
official baselines, and it is about 20 lines. Everything a team does afterwards is an
improvement on this, not a replacement for it.
"""
import numpy as np
import pandas as pd

DATA = "/data/cureatime"          # change to "data/processed" if you run it locally

# 1. load
X_train = pd.read_csv(f"{DATA}/X_train.csv.gz", index_col=0)
y_train = pd.read_csv(f"{DATA}/y_train.csv", index_col=0)["moa"]
X_test = pd.read_csv(f"{DATA}/X_test.csv.gz", index_col=0)
meta_test = pd.read_csv(f"{DATA}/meta_test.csv", index_col=0)

# 2. one average fingerprint per mechanism class
profiles = X_train.groupby(y_train).mean()

# 3. cosine similarity = do two fingerprints point the same way?
unit = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
similarity = pd.DataFrame(unit(X_test[profiles.columns].values) @ unit(profiles.values).T,
                          index=X_test.index, columns=profiles.index)

# 4. a drug was measured at several doses: average them into one row per drug
submission = similarity.groupby(meta_test["compound"]).mean()
submission.index.name = "compound"

# 5. save
submission.reset_index().to_csv("submission.csv", index=False)
print(f"wrote submission.csv for {len(submission)} drugs")
print(submission.idxmax(axis=1).head(10))
