#!/usr/bin/env python3
"""Train a >50K income classifier and build a segmentation model.

The script prefers the local `census-bureau.data` file. If the file is a Git LFS
pointer (common in take-home repos) it falls back to OpenML's Adult dataset so
that the pipeline remains executable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = Path("census-bureau.data")
COLUMNS_FILE = Path("census-bureau.columns")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return True
    head = path.read_text(errors="ignore")[:256]
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def load_dataset() -> Tuple[pd.DataFrame, str, str]:
    """Load dataset and return (dataframe, target_column, weight_column)."""
    if DATA_FILE.exists() and COLUMNS_FILE.exists() and not _is_lfs_pointer(DATA_FILE):
        columns = [c.strip() for c in COLUMNS_FILE.read_text().splitlines() if c.strip()]
        df = pd.read_csv(DATA_FILE, header=None, names=columns)
        target_col = "income"
        if target_col not in df.columns:
            # last column in the provided CPS file is the income label
            target_col = df.columns[-1]
        weight_col = "instance weight"
        if weight_col not in df.columns:
            # fallback to a likely weight column name
            weight_candidates = [c for c in df.columns if "weight" in c.lower()]
            weight_col = weight_candidates[0] if weight_candidates else "_weight"
            if weight_col == "_weight":
                df[weight_col] = 1.0
        return df, target_col, weight_col

    # Fallback: Adult dataset from OpenML with conceptually similar columns.
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name="adult", version=2, as_frame=True)
    df = bunch.frame.copy()
    df["instance weight"] = 1.0
    return df, "class", "instance weight"


def make_preprocessor(df: pd.DataFrame, target_col: str, weight_col: str) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in {target_col, weight_col}]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def classification_task(df: pd.DataFrame, target_col: str, weight_col: str) -> dict:
    preprocessor, _, _ = make_preprocessor(df, target_col, weight_col)

    X = df.drop(columns=[target_col])
    y_raw = df[target_col].astype(str).str.strip()
    y = y_raw.str.contains(r">50K|50000\+", regex=True)
    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        weights,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(X_train, y_train, model__sample_weight=w_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred, sample_weight=w_test),
        "precision": precision_score(y_test, y_pred, sample_weight=w_test, zero_division=0),
        "recall": recall_score(y_test, y_pred, sample_weight=w_test, zero_division=0),
        "f1": f1_score(y_test, y_pred, sample_weight=w_test, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob, sample_weight=w_test),
        "n_rows": int(df.shape[0]),
        "positive_rate": float(np.average(y, weights=weights)),
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "classification_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def segmentation_task(df: pd.DataFrame, target_col: str, weight_col: str) -> dict:
    preprocessor, numeric_cols, categorical_cols = make_preprocessor(df, target_col, weight_col)
    X = df.drop(columns=[target_col])
    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)

    X_enc = preprocessor.fit_transform(X)

    pca = PCA(n_components=0.9, random_state=RANDOM_STATE)
    X_dense = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc
    X_red = pca.fit_transform(X_dense)

    best_k = 3
    best_score = -1.0
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_red)
        score = silhouette_score(X_red, labels)
        if score > best_score:
            best_score = score
            best_k = k

    final_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=30)
    cluster = final_kmeans.fit_predict(X_red)

    profile = df.copy()
    profile["cluster"] = cluster

    segment_summary: dict[str, dict] = {}
    for c in sorted(profile["cluster"].unique()):
        seg = profile[profile["cluster"] == c]
        w = pd.to_numeric(seg[weight_col], errors="coerce").fillna(1.0)
        record = {
            "weighted_share": float(w.sum() / weights.sum()),
            "size": int(seg.shape[0]),
        }

        for col in numeric_cols[:5]:
            vals = pd.to_numeric(seg[col], errors="coerce")
            vals = vals.fillna(vals.median())
            record[f"avg_{col}"] = float(np.average(vals, weights=w))

        for col in categorical_cols[:3]:
            top = (
                seg.groupby(col, dropna=False)[weight_col]
                .sum()
                .sort_values(ascending=False)
                .head(2)
                .index.astype(str)
                .tolist()
            )
            record[f"top_{col}"] = top

        segment_summary[f"segment_{c}"] = record

    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "segmentation_summary.json").open("w") as f:
        json.dump(
            {
                "k": best_k,
                "silhouette": best_score,
                "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
                "segments": segment_summary,
            },
            f,
            indent=2,
        )

    return {
        "k": best_k,
        "silhouette": best_score,
        "segments": segment_summary,
    }


def main() -> None:
    df, target_col, weight_col = load_dataset()
    cls_metrics = classification_task(df, target_col, weight_col)
    seg_metrics = segmentation_task(df, target_col, weight_col)

    print("Classification metrics:")
    for k, v in cls_metrics.items():
        print(f"  {k}: {v}")

    print("\nSegmentation:")
    print(f"  clusters: {seg_metrics['k']}")
    print(f"  silhouette: {seg_metrics['silhouette']:.4f}")


if __name__ == "__main__":
    main()
