"""Filtering and heuristic labeling for IT developer resumes.

This module implements:
- Strict filtering of IT developer resumes based on position titles.
- Heuristic target creation: ``dev_level`` in {junior, middle, senior}.

The task is PoC-oriented. Rules are intentionally simple and explainable.
"""

from __future__ import annotations

import re
from typing import Final

import numpy as np
import pandas as pd

DEV_INCLUDE_PATTERNS: Final[list[str]] = [
    r"\bdeveloper\b",
    r"\bsoftware\b",
    r"\bprogrammer\b",
    r"\bразработ",
    r"\bпрограммист\b",
    r"\bинженер[- ]?программист\b",
    r"\bпрограммист[- ]?разработчик\b",
    r"\bweb[- ]?(developer|разработчик|программист)\b",
    r"\bвеб[- ]?(разработчик|программист)\b",
    r"\bbackend\b",
    r"\bfrontend\b",
    r"\bfront[- ]?end\b",
    r"\bfull[- ]?stack\b",
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

DEV_EXCLUDE_PATTERNS: Final[list[str]] = [
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

SENIOR_TITLE_PATTERNS: Final[list[str]] = [
    r"\bsenior\b",
    r"\bсеньор\b|\bсениор\b",
    r"\blead\b",
    r"\bteam lead\b|\bтимлид\b|\bтим лид\b",
    r"\btech lead\b",
    r"\bprincipal\b|\bstaff\b",
    r"\bархитектор\b",
    r"\bстарш(?:ий|ая)\b",
]

MIDDLE_TITLE_PATTERNS: Final[list[str]] = [
    r"\bmiddle\b",
    r"\bмидл\b|\bмиддл\b",
    r"\bregular\b",
    r"\bведущ(?:ий|ая)\b",
]

JUNIOR_TITLE_PATTERNS: Final[list[str]] = [
    r"\bjunior\b",
    r"\bджуниор\b|\bджун\b",
    r"\bintern\b|\bстажер\b|\bстажёр\b|\btrainee\b",
    r"\bмладш",
    r"\bначинающ",
]


def normalize_text(value: object) -> str:
    """Convert arbitrary values into normalized lowercase text.

    Args:
        value (object): Input value.

    Returns:
        str: Normalized string (lowercase, trimmed). Empty string for missing values.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def build_position_text(dataframe: pd.DataFrame) -> pd.Series:
    """Build a combined text field from ``position`` and ``last_position``.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        pd.Series: Combined lowercase position text.

    Raises:
        KeyError: If required columns are missing.
    """
    if "position" not in dataframe.columns or "last_position" not in dataframe.columns:
        raise KeyError("Expected 'position' and 'last_position' columns in input dataframe.")

    position = dataframe["position"].map(normalize_text)
    last_position = dataframe["last_position"].map(normalize_text)
    return (position + " " + last_position).str.strip()


def is_it_developer(position_text: str) -> bool:
    """Determine whether a resume belongs to an IT developer (strict heuristic).

    Args:
        position_text (str): Combined position text.

    Returns:
        bool: True if matches developer include patterns and does not match exclude patterns.
    """
    has_include = any(re.search(pattern, position_text) for pattern in DEV_INCLUDE_PATTERNS)
    has_exclude = any(re.search(pattern, position_text) for pattern in DEV_EXCLUDE_PATTERNS)
    return bool(has_include and not has_exclude)


def label_level(position_text: str, experience_years: object) -> str | None:
    """Assign a level label junior/middle/senior using title markers and experience fallback.

    Args:
        position_text (str): Combined position text.
        experience_years (object): Experience value (expected numeric-like).

    Returns:
        str | None: One of {"junior", "middle", "senior"} or None if cannot label.

    Heuristic priority:
    1) Explicit title markers.
    2) Fallback by experience thresholds:
       - exp < 2  -> junior
       - exp < 8  -> middle
       - exp >= 8 -> senior
    """
    if any(re.search(pattern, position_text) for pattern in SENIOR_TITLE_PATTERNS):
        return "senior"
    if any(re.search(pattern, position_text) for pattern in MIDDLE_TITLE_PATTERNS):
        return "middle"
    if any(re.search(pattern, position_text) for pattern in JUNIOR_TITLE_PATTERNS):
        return "junior"

    try:
        experience = float(experience_years)
    except (TypeError, ValueError):
        return None

    if np.isnan(experience):
        return None

    if experience < 2.0:
        return "junior"
    if experience < 8.0:
        return "middle"
    return "senior"


def filter_it_developers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filter strict IT developer resumes and add ``position_text`` column.

    Args:
        dataframe (pd.DataFrame): Source dataset.

    Returns:
        pd.DataFrame: Subset containing IT developers only.
    """
    result = dataframe.copy()
    result["position_text"] = build_position_text(result)
    mask = result["position_text"].apply(is_it_developer)
    return result[mask].copy()


def add_level_label(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add ``dev_level`` column to a developer dataset.

    Args:
        dataframe (pd.DataFrame): IT developer subset (ideally output of filter_it_developers).

    Returns:
        pd.DataFrame: Labeled dataset with column ``dev_level``. Rows with missing label are dropped.
    """
    result = dataframe.copy()
    if "position_text" not in result.columns:
        result["position_text"] = build_position_text(result)

    result["dev_level"] = result.apply(
        lambda row: label_level(row["position_text"], row.get("experience_years")),
        axis=1,
    )

    return result.dropna(subset=["dev_level"]).copy()
