"""Feature engineering and preprocessing for developer level classification.

This module builds:
- Tabular features from prepared resume data
- Light-weight "skill flags" extracted from ``position_text`` (proxy for skills/stack)
- A scikit-learn ColumnTransformer for preprocessing:
  - scaling numeric features
  - one-hot encoding categorical features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

POSITION_FLAG_PATTERNS: Final[dict[str, str]] = {
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


@dataclass(frozen=True)
class FeatureSpec:
    """A specification of which columns are used as numeric/categorical features."""

    numeric_features: list[str]
    categorical_features: list[str]


DEFAULT_FEATURES: Final[FeatureSpec] = FeatureSpec(
    numeric_features=[
        "age",
        "salary",
        "experience_years",
        "education_last_year",
        "has_car",
        "relocation",
        "has_master",
        *list(POSITION_FLAG_PATTERNS.keys()),
    ],
    categorical_features=[
        "city",
        "education_level",
        "business_trips",
        "schedule",
    ],
)


def add_position_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add binary "skill flag" features derived from ``position_text``.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Copy of input dataframe with added ``pos_*`` columns.

    Notes:
        These flags are a cheap proxy for skills/stack without heavy NLP (TF-IDF).
    """
    if "position_text" not in dataframe.columns:
        return dataframe

    result = dataframe.copy()
    text = result["position_text"].fillna("").astype(str).str.lower()

    for column_name, pattern in POSITION_FLAG_PATTERNS.items():
        # pandas uses regex by default; patterns here are already regex-based.
        result[column_name] = text.str.contains(pattern, regex=True, na=False).astype(int)

    return result


def make_preprocessor(feature_spec: FeatureSpec) -> ColumnTransformer:
    """Create a scikit-learn preprocessor for numeric and categorical features.

    Args:
        feature_spec (FeatureSpec): Feature column lists.

    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), feature_spec.numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), feature_spec.categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_features_and_target(
        dataframe: pd.DataFrame,
        feature_spec: FeatureSpec,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build model matrix X and target vector y from the labeled dataset.

    Args:
        dataframe (pd.DataFrame): Labeled dataframe that includes ``dev_level`` and feature columns.
        feature_spec (FeatureSpec): Feature definition.

    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y)

    Raises:
        KeyError: If required columns are missing.
    """
    enriched_dataframe = add_position_flags(dataframe)

    required_columns = ["dev_level"] + feature_spec.numeric_features + feature_spec.categorical_features
    missing_columns = [col for col in required_columns if col not in enriched_dataframe.columns]
    if missing_columns:
        raise KeyError("Missing required columns: " + ", ".join(missing_columns))

    features = enriched_dataframe[feature_spec.numeric_features + feature_spec.categorical_features].copy()
    target = enriched_dataframe["dev_level"].astype(str)

    for column_name in feature_spec.numeric_features:
        features[column_name] = pd.to_numeric(features[column_name], errors="coerce")
        if features[column_name].isna().any():
            features[column_name] = features[column_name].fillna(features[column_name].median())

    for column_name in feature_spec.categorical_features:
        features[column_name] = features[column_name].fillna("UNKNOWN").astype(str)

    return features, target
