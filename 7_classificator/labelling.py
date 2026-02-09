from __future__ import annotations

import re
from typing import Final

import numpy as np
import pandas as pd


def _normalize_text(value: object) -> str:
    """Normalize text field to a safe lowercase string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def join_position_text(df: pd.DataFrame) -> pd.Series:
    """Join position and last_position into one normalized text column."""
    if "position" not in df.columns or "last_position" not in df.columns:
        raise KeyError("Expected 'position' and 'last_position' columns in input dataframe.")
    return (df["position"].map(_normalize_text) + " " + df["last_position"].map(_normalize_text)).str.strip()


# ---------------------------------------------------------------------------
# Strict IT developer filter (high precision, lower recall)
# ---------------------------------------------------------------------------

_INCLUDE_PATTERNS: Final[list[str]] = [
    # generic dev words
    r"\bdeveloper\b",
    r"\bsoftware\b",
    r"\bprogrammer\b",
    r"\bразработ",
    r"\bпрограммист\b",
    r"\bинженер[- ]?программист\b",
    r"\bпрограммист[- ]?разработчик\b",
    r"\bweb[- ]?(developer|разработчик|программист)\b",
    r"\bвеб[- ]?(разработчик|программист)\b",
    # directions
    r"\bbackend\b",
    r"\bfrontend\b",
    r"\bfront[- ]?end\b",
    r"\bfull[- ]?stack\b",
    # tech keywords
    r"\bpython\b",
    r"\bjava\b",
    r"\bjavascript\b|\bjs\b",
    r"\bphp\b",
    r"\bgolang\b|\bgo\b",
    r"\bc\+\+\b",
    r"\bc#\b",
    r"\b\.net\b|\bdotnet\b",
    r"\bios\b|\bandroid\b",
    r"\b1c\b|\b1с\b",
]

_EXCLUDE_PATTERNS: Final[list[str]] = [
    # non-dev / ambiguous IT roles that we intentionally drop for "strict dev"
    r"\bqa\b",
    r"\bтестир",
    r"\banalyst\b|\bаналитик\b",
    r"\bproduct\b|\bproject\b|\bpm\b|\bменеджер\b",
    r"\bdesigner\b|\bдизайнер\b",
    r"\bsupport\b|\bподдержк\b",
    r"\bадминистратор\b|\bsysadmin\b|\bsystem administrator\b",
    r"\bdevops\b",
    r"\bdata scientist\b|\bдата саентист\b",
]


def is_it_developer(position_text: str) -> bool:
    """Keep only resumes that look like software developers (strict heuristic)."""
    has_include = any(re.search(pat, position_text) for pat in _INCLUDE_PATTERNS)
    has_exclude = any(re.search(pat, position_text) for pat in _EXCLUDE_PATTERNS)
    return bool(has_include and not has_exclude)


# ---------------------------------------------------------------------------
# Level labeling: title first, fallback to experience
# ---------------------------------------------------------------------------

_SENIOR_PATTERNS: Final[list[str]] = [
    r"\bsenior\b",
    r"\bсеньор\b|\bсениор\b",
    r"\blead\b",
    r"\bteam lead\b|\bтимлид\b|\bтим лид\b",
    r"\btech lead\b",
    r"\bprincipal\b|\bstaff\b",
    r"\bархитектор\b",
    r"\bстарш(?:ий|ая)\b",  # старший разработчик/программист
]

_MIDDLE_PATTERNS: Final[list[str]] = [
    r"\bmiddle\b",
    r"\bмидл\b|\bмиддл\b",
    r"\bregular\b",
    r"\bведущ(?:ий|ая)\b",  # считаем как middle+ (часто не равен senior)
]

_JUNIOR_PATTERNS: Final[list[str]] = [
    r"\bjunior\b",
    r"\bджуниор\b|\bджун\b",
    r"\bintern\b|\bстажер\b|\bстажёр\b|\btrainee\b",
    r"\bмладш",
    r"\bначинающ",
]


def label_level(position_text: str, experience_years: object) -> str | None:
    """Hybrid labeling: title markers first, fallback to experience.

    Fallback thresholds are chosen to:
    - increase coverage (drop fewer rows)
    - avoid turning most of the dataset into "senior"
      (note: in hh resumes many people have large experience)

    Rules:
    - explicit title markers override experience
    - else fallback by experience:
        exp < 2 -> junior
        2 <= exp < 8 -> middle
        exp >= 8 -> senior
    """
    if any(re.search(pat, position_text) for pat in _SENIOR_PATTERNS):
        return "senior"
    if any(re.search(pat, position_text) for pat in _MIDDLE_PATTERNS):
        return "middle"
    if any(re.search(pat, position_text) for pat in _JUNIOR_PATTERNS):
        return "junior"

    try:
        exp = float(experience_years)
    except (TypeError, ValueError):
        return None

    if np.isnan(exp):
        return None

    if exp < 2.0:
        return "junior"
    if exp < 8.0:
        return "middle"
    return "senior"


def filter_it(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only IT developers (strict heuristic by position text)."""
    df_out = df.copy()
    df_out["position_text"] = join_position_text(df_out)
    mask_it = df_out["position_text"].apply(is_it_developer)
    return df_out[mask_it].copy()


def add_level_label(df_it: pd.DataFrame) -> pd.DataFrame:
    """Add dev_level label and drop rows where it cannot be inferred."""
    df_out = df_it.copy()
    if "position_text" not in df_out.columns:
        df_out["position_text"] = join_position_text(df_out)

    df_out["dev_level"] = df_out.apply(
        lambda row: label_level(row["position_text"], row.get("experience_years")),
        axis=1,
    )
    return df_out.dropna(subset=["dev_level"]).copy()
