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


# --- IT developer filter ------------------------------------------------------
# Goal: catch developers + engineering managers in IT, even if title is generic like "руководитель отдела".
# Trade-off: heuristic, may include some non-dev IT roles (acceptable for PoC).

_DEV_SIGNAL_PATTERNS: Final[list[str]] = [
    # English dev signals
    r"\bdeveloper\b",
    r"\bsoftware\b",
    r"\bprogrammer\b",
    r"\bbackend\b",
    r"\bfrontend\b",
    r"\bfront[- ]?end\b",
    r"\bfull[- ]?stack\b",
    r"\bweb\b",
    # Tech keywords (optional, but helpful)
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
    # Russian dev signals
    r"\bразработ",
    r"\bпрограммист",
    r"\bинженер\b",
    r"\bвеб[- ]?разработ",
    r"\bинженер[- ]?программист",
]

_MANAGER_SIGNAL_PATTERNS: Final[list[str]] = [
    r"\bteam lead\b|\bтимлид\b|\bтим лид\b",
    r"\btech lead\b",
    r"\blead\b",
    r"\bруководител",
    r"\bначальник",
    r"\bhead\b",
    r"\bcto\b",
    r"\bдиректор\b.*\bit\b|\bit\b.*\bдиректор\b",
    r"\bруководител\b.*\bit\b|\bit\b.*\bруководител\b",
    r"\bруководител\b.*\bразработ",
    r"\bруководител\b.*\bинженер",
]

# Strong non-IT / non-dev exclusions. Keep this list focused to avoid throwing away legitimate roles.
_NON_IT_EXCLUDE_PATTERNS: Final[list[str]] = [
    r"\bhr\b|\bрекрутер\b|\bкадров\b",
    r"\bбухгалтер\b|\baccountant\b",
    r"\bюрист\b|\blawyer\b",
    r"\bпродавец\b|\bsales\b",
    r"\bмаркетолог\b|\bmarketing\b",
    r"\bофис[- ]?менеджер\b",
    r"\bводитель\b",
    r"\bкладовщик\b",
    r"\bмедсестра\b|\bврач\b",
    r"\bучител\b|\bпреподавател\b",
]

# Optional: still exclude obvious non-developer IT roles if you want "developers only".
# If you want to keep managers of IT-infra, remove sysadmin/devops exclusions.
_NON_DEV_IT_EXCLUDE_PATTERNS: Final[list[str]] = [
    r"\bqa\b|\bтестир",
    r"\bsupport\b|\bподдержк\b",
    r"\bадминистратор\b|\bsysadmin\b|\bsystem administrator\b",
    r"\bdesigner\b|\bдизайнер\b",
    r"\bproduct\b|\bproject\b|\bpm\b|\bменеджер\b(?!\s*по\s*разработ)",  # keep "manager of development" patterns
]


def _has_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pat, text) for pat in patterns)

def has_dev_signal(position_text: str) -> bool:
    """Return True if title contains clear developer/engineering signals."""
    return _has_any(_DEV_SIGNAL_PATTERNS, position_text)


def has_it_anchor(position_text: str) -> bool:
    """Return True if title contains generic IT anchors (for managers/heads)."""
    it_anchor_patterns = [
        r"\bit\b",
        r"\bинформационн",
        r"\bразработ",
        r"\bsoftware\b",
        r"\bengineering\b",
        r"\bразработка\b",
        r"\bразработки\b",
    ]
    return _has_any(it_anchor_patterns, position_text)

def is_it_developer_or_manager(position_text: str) -> bool:
    """Broader heuristic: keep dev roles and engineering managers likely related to IT development."""
    text = position_text

    # First, drop clear non-IT roles
    if _has_any(_NON_IT_EXCLUDE_PATTERNS, text):
        return False

    has_manager_signal = _has_any(_MANAGER_SIGNAL_PATTERNS, text)

    # Exclude obvious non-dev IT roles only if there is no manager/dev signal.
    # This prevents dropping e.g. "руководитель devops" if you consider it part of IT.
    if _has_any(_NON_DEV_IT_EXCLUDE_PATTERNS, text) and not has_manager_signal and not has_dev_signal(text):
        return False

    return bool(has_dev_signal(text) or has_manager_signal)


# --- Level labeling -----------------------------------------------------------

_SENIOR_PATTERNS: Final[list[str]] = [
    r"\bsenior\b",
    r"\bсеньор\b",
    r"\bсениор\b",
    r"\blead\b",
    r"\bteam lead\b|\bтимлид\b|\bтим лид\b",
    r"\btech lead\b",
    r"\bprincipal\b|\bstaff\b",
    r"\bархитектор\b",

]

_MIDDLE_PATTERNS: Final[list[str]] = [
    r"\bmiddle\b",
    r"\bмидл\b",
    r"\bмиддл\b",
    r"\bregular\b",
    r"\bведущ(ий|ая)\b",
]

_JUNIOR_PATTERNS: Final[list[str]] = [
    r"\bjunior\b",
    r"\bджуниор\b",
    r"\bджун\b",
    r"\bintern\b|\bстажер\b|\bстажёр\b|\btrainee\b",
    r"\bмладш",
    r"\bначинающ",
]


def label_level_from_title(position_text: str) -> str | None:
    """Label dev level from title keywords only."""
    if any(re.search(pat, position_text) for pat in _SENIOR_PATTERNS):
        return "senior"
    if any(re.search(pat, position_text) for pat in _MIDDLE_PATTERNS):
        return "middle"
    if any(re.search(pat, position_text) for pat in _JUNIOR_PATTERNS):
        return "junior"
    return None


def label_level_hybrid(position_text: str, experience_years: object) -> str | None:
    """Hybrid labeling: title first, fallback to experience_years with stricter rules.

    Fallback is applied only for titles that look IT-related enough.
    """
    from_title = label_level_from_title(position_text)
    if from_title is not None:
        return from_title

    # Fallback by experience only if the title has clear dev signal OR IT anchor.
    # This prevents turning every "engineer/manager" into senior by experience.
    if not (has_dev_signal(position_text) or has_it_anchor(position_text)):
        return None

    try:
        exp = float(experience_years)
    except (TypeError, ValueError):
        return None

    if np.isnan(exp):
        return None

    if exp < 1.5:
        return "junior"
    if exp < 7.0:
        return "middle"
    return "senior"


def filter_it(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only IT dev/engineering-manager resumes (heuristic by position text)."""
    df_out = df.copy()
    df_out["position_text"] = join_position_text(df_out)
    mask_it = df_out["position_text"].apply(is_it_developer_or_manager)
    return df_out[mask_it].copy()


def add_level_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add dev_level label and drop unlabeled rows."""
    df_out = df.copy()
    if "position_text" not in df_out.columns:
        df_out["position_text"] = join_position_text(df_out)

    # Hybrid labeling to keep more data
    df_out["dev_level"] = df_out.apply(
        lambda row: label_level_hybrid(row["position_text"], row.get("experience_years")),
        axis=1,
    )

    return df_out.dropna(subset=["dev_level"]).copy()
