from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSpec:
    numeric_features: list[str]
    categorical_features: list[str]


DEFAULT_FEATURES = FeatureSpec(
    numeric_features=[
        "age",
        "salary",
        "experience_years",
        "has_car",
        "relocation",
        "has_master",
    ],
    categorical_features=[
        "city",
        "education_level",
        "business_trips",
        "schedule",
    ],
)


def make_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """Build sklearn preprocessing pipeline."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), spec.numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), spec.categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_xy(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X and y from the labeled dataframe."""
    required = ["dev_level"] + spec.numeric_features + spec.categorical_features
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))

    X = df[spec.numeric_features + spec.categorical_features].copy()
    y = df["dev_level"].astype(str)

    # Minimal cleanup
    for col in spec.numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    for col in spec.categorical_features:
        X[col] = X[col].fillna("UNKNOWN").astype(str)

    return X, y
