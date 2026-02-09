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


_INCLUDE_PATTERNS: Final[list[str]] = [
    r"\bdeveloper\b",
    r"\bsoftware\b",
    r"\bengineer\b",
    r"\bprogrammer\b",
    r"\bbackend\b",
    r"\bfrontend\b",
    r"\bfront[- ]?end\b",
    r"\bfull[- ]?stack\b",
    r"\bweb\b",
    r"\bpython\b",
    r"\bjava\b",
    r"\bjavascript\b",
    r"\bphp\b",
    r"\bgolang\b|\bgo\b",
    r"\bc\+\+\b",
    r"\bc#\b",
    r"\b\.net\b",
    r"\bios\b|\bandroid\b",
    r"\b1c\b|\b1с\b",
    r"\bразработ",
    r"\bпрограммист",
    r"\bинженер[- ]?программист",
    r"\bвеб[- ]?разработ",
]

_EXCLUDE_PATTERNS: Final[list[str]] = [
    r"\bqa\b",
    r"\bтестир",
    r"\banalyst\b|\bаналитик\b",
    r"\bpm\b|\bproduct\b|\bproject\b|\bменеджер\b",
    r"\bdesigner\b|\bдизайнер\b",
    r"\bsupport\b|\bподдержк\b",
    r"\bадминистратор\b|\bsysadmin\b|\bsystem administrator\b",
    r"\bdevops\b",
    r"\bdata scientist\b|\bдата саентист\b",
]


def is_it_developer(position_text: str) -> bool:
    """Heuristic filter: keep resumes that look like software developers."""
    has_include = any(re.search(pat, position_text) for pat in _INCLUDE_PATTERNS)
    has_exclude = any(re.search(pat, position_text) for pat in _EXCLUDE_PATTERNS)
    return bool(has_include and not has_exclude)


_SENIOR_PATTERNS: Final[list[str]] = [
    r"\bsenior\b",
    r"\bсеньор\b",
    r"\blead\b",
    r"\bteam lead\b|\bтимлид\b|\bтим лид\b",
    r"\btech lead\b",
    r"\bprincipal\b|\bstaff\b",
    r"\bархитектор\b",
    r"\bведущ(ий|ая)\b",
]

_MIDDLE_PATTERNS: Final[list[str]] = [
    r"\bmiddle\b",
    r"\bмидл\b",
    r"\bregular\b",
]

_JUNIOR_PATTERNS: Final[list[str]] = [
    r"\bjunior\b",
    r"\bджуниор\b",
    r"\bintern\b|\bстажер\b|\bстажёр\b|\btrainee\b",
    r"\bмладш",
    r"\bначинающ",
]


def label_level(position_text: str) -> str | None:
    """Heuristic labeling of developer level from position text."""
    if any(re.search(pat, position_text) for pat in _SENIOR_PATTERNS):
        return "senior"
    if any(re.search(pat, position_text) for pat in _MIDDLE_PATTERNS):
        return "middle"
    if any(re.search(pat, position_text) for pat in _JUNIOR_PATTERNS):
        return "junior"
    return None


def filter_it(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only IT developer resumes (heuristic by position text)."""
    df_out = df.copy()
    df_out["position_text"] = join_position_text(df_out)
    mask_it = df_out["position_text"].apply(is_it_developer)
    return df_out[mask_it].copy()


def add_level_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add dev_level label and drop unlabeled rows."""
    df_out = df.copy()
    if "position_text" not in df_out.columns:
        df_out["position_text"] = join_position_text(df_out)

    df_out["dev_level"] = df_out["position_text"].apply(label_level)
    return df_out.dropna(subset=["dev_level"]).copy()
