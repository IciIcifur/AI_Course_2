from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_class_balance(y: pd.Series, output_path: Path, title: str) -> None:
    """Plot class distribution as a bar chart."""
    counts = y.value_counts().reindex(["junior", "middle", "senior"])
    counts = counts.fillna(0).astype(int)

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index, counts.values)
    plt.title(title)
    plt.xlabel("Level")
    plt.ylabel("Number of resumes")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
