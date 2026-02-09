from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Simple "skills" proxy extracted from position_text
POSITION_FLAG_PATTERNS: dict[str, str] = {
    "pos_java": r"\bjava\b",
    "pos_python": r"\bpython\b|\bпитон\b",
    "pos_js": r"\bjavascript\b|\bjs\b",
    "pos_php": r"\bphp\b",
    "pos_frontend": r"\bfrontend\b|\bfront[- ]?end\b|\bфронтенд\b",
    "pos_backend": r"\bbackend\b|\bбэкенд\b|\bбекенд\b",
    "pos_fullstack": r"\bfull[- ]?stack\b|\bфуллстек\b",
    "pos_mobile": r"\bios\b|\bandroid\b|\bmobile\b|\bмобил",
    "pos_1c": r"\b1c\b|\b1с\b",
    "pos_dotnet": r"\b\.net\b|\bdotnet\b",
    "pos_csharp": r"\bc#\b",
    "pos_cpp": r"\bc\+\+\b",
}


def add_position_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary features from position_text (cheap proxy for stack/skills)."""
    if "position_text" not in df.columns:
        return df

    out = df.copy()
    text = out["position_text"].fillna("").astype(str).str.lower()
    for col, pat in POSITION_FLAG_PATTERNS.items():
        out[col] = text.str.contains(pat, regex=True, na=False).astype(int)
    return out


@dataclass(frozen=True)
class FeatureSpec:
    numeric_features: list[str]
    categorical_features: list[str]


DEFAULT_FEATURES = FeatureSpec(
    numeric_features=[
        "age",
        "salary",
        "experience_years",
        "education_last_year",
        "has_car",
        "relocation",
        "has_master",
        # position flags
        *list(POSITION_FLAG_PATTERNS.keys()),
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
    df = add_position_flags(df)

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
