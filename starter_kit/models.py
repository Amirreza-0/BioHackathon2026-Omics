"""
Model zoo. Every model has .fit(X, y) and .predict_proba(X) -> DataFrame (rows x classes).

  centroid  cosine similarity to each class's mean signature (CMap-style "connectivity").
            Fully interpretable: the class centroids ARE the mechanism signatures.
  knn       cosine k-nearest neighbours over training signatures.
  logreg    standardize + multinomial logistic regression (class-balanced).
  rf        random forest (class-balanced).
  mlp       small neural network (1 hidden layer, early stopping).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED


def _unit(X):
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _softmax(S, temp):
    S = (S - S.max(axis=1, keepdims=True)) / temp
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


class CentroidModel:
    def __init__(self, temp=0.1):
        self.temp = temp

    def fit(self, X, y):
        self.classes_ = np.array(sorted(pd.unique(y)))
        U = _unit(X)
        self.centroids_ = np.vstack([U[np.asarray(y) == c].mean(axis=0) for c in self.classes_])
        self.centroids_ = _unit(self.centroids_)
        self.genes_ = list(X.columns)
        return self

    def predict_proba(self, X):
        S = _unit(X[self.genes_]) @ self.centroids_.T          # cosine similarity
        return pd.DataFrame(_softmax(S, self.temp), index=X.index, columns=self.classes_)


class KNNModel:
    def __init__(self, k=7, temp=0.1):
        self.k, self.temp = k, temp

    def fit(self, X, y):
        self.U_ = _unit(X)
        self.y_ = np.asarray(y)
        self.classes_ = np.array(sorted(pd.unique(y)))
        self.genes_ = list(X.columns)
        return self

    def predict_proba(self, X):
        S = _unit(X[self.genes_]) @ self.U_.T
        k = min(self.k, S.shape[1])
        idx = np.argsort(-S, axis=1)[:, :k]
        P = np.zeros((len(X), len(self.classes_)))
        cpos = {c: i for i, c in enumerate(self.classes_)}
        for r in range(len(X)):
            w = np.exp(S[r, idx[r]] / self.temp)
            for j, wj in zip(idx[r], w):
                P[r, cpos[self.y_[j]]] += wj
        P /= P.sum(axis=1, keepdims=True)
        return pd.DataFrame(P, index=X.index, columns=self.classes_)


class SklearnModel:
    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        # integer-encode labels (sklearn's MLP early stopping fails on string labels)
        self.classes_ = np.array(sorted(pd.unique(y)))
        codes = pd.Categorical(np.asarray(y), categories=self.classes_).codes
        self.est.fit(X.values, codes)
        self.genes_ = list(X.columns)
        return self

    def predict_proba(self, X):
        P = self.est.predict_proba(X[self.genes_].values)
        cols = self.classes_[self.est.classes_]               # map codes back to class names
        return pd.DataFrame(P, index=X.index, columns=cols).reindex(columns=self.classes_, fill_value=0.0)


def make_model(name: str):
    if name == "centroid":
        return CentroidModel()
    if name == "knn":
        return KNNModel()
    if name == "logreg":
        return SklearnModel(make_pipeline(StandardScaler(), LogisticRegression(
            C=0.05, max_iter=3000, class_weight="balanced")))
    if name == "rf":
        return SklearnModel(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED))
    if name == "mlp":
        return SklearnModel(make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True,
            validation_fraction=0.15, max_iter=500, random_state=SEED)))
    raise ValueError(name)


MODELS = ["centroid", "knn", "logreg", "rf", "mlp"]
