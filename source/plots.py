"""Visualization utilities for the classification PoC."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LEVEL_ORDER = ["junior", "middle", "senior"]


def plot_class_balance(y: pd.Series, output_path: Path, title: str) -> None:
    """Plot class distribution and save it to disk.

    Args:
        y (pd.Series): Target labels.
        output_path (Path): Where to save the plot.
        title (str): Plot title.
    """
    class_counts = y.value_counts().reindex(LEVEL_ORDER).fillna(0).astype(int)

    plt.figure(figsize=(7, 4))
    plt.bar(class_counts.index, class_counts.values)
    plt.title(title)
    plt.xlabel("Level")
    plt.ylabel("Number of resumes")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
