"""Train a PoC classifier for developer level (junior/middle/senior) from HH.ru resumes.

The pipeline:
1) Load the prepared dataset (CSV).
2) Filter IT developer resumes using heuristic rules based on position titles.
3) Build the target variable ``dev_level`` using title markers and experience fallback.
4) Plot class balance.
5) Train a baseline model (most frequent class) and a LinearSVC classifier.
6) Print a classification report and balanced accuracy on a hold-out split.

Notes:
- Labels are heuristic, so metrics reflect agreement with the heuristic "ground truth".
- The goal is a proof of concept (PoC), not a production-grade model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from features import DEFAULT_FEATURES, build_features_and_target, make_preprocessor
from labelling import add_level_label, filter_it_developers
from plots import plot_class_balance

DEFAULT_INPUT_PATH: Final[Path] = Path("data/processed/hh_prepared_raw.csv")
DEFAULT_OUTPUT_DIR: Final[Path] = Path("resources")
DEFAULT_TEST_SIZE: Final[float] = 0.2
DEFAULT_RANDOM_STATE: Final[int] = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train PoC classifier for junior/middle/senior developer level."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to prepared raw CSV (from 5_parsing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of data to use as validation set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for train/validation split.",
    )
    return parser.parse_args()


def ensure_file_exists(file_path: Path, description: str) -> None:
    """Ensure that a file exists.

    Args:
        file_path (Path): Path to check.
        description (str): Human-readable description for error messages.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"{description} not found: {file_path}")


def print_filtering_summary(
        total_rows: int,
        it_rows: int,
        labeled_rows: int,
) -> None:
    """Print dataset sizes after each processing step.

    Args:
        total_rows (int): Total number of rows in the source dataset.
        it_rows (int): Rows after IT developer filtering.
        labeled_rows (int): Rows after target labeling.
    """
    dropped_after_it = total_rows - it_rows
    dropped_after_labeling = it_rows - labeled_rows

    dropped_after_it_pct = dropped_after_it / max(total_rows, 1) * 100.0
    dropped_after_labeling_pct = dropped_after_labeling / max(it_rows, 1) * 100.0

    print("=== Filtering summary ===")
    print(f"Total rows in prepared_raw: {total_rows}")
    print(
        "After IT developer filter:  "
        f"{it_rows} (dropped {dropped_after_it}, {dropped_after_it_pct:.1f}%)"
    )
    print(
        "After level labeling:       "
        f"{labeled_rows} (dropped {dropped_after_labeling}, {dropped_after_labeling_pct:.1f}%)"
    )
    print()


def main() -> None:
    """Run training and evaluation."""
    args = parse_args()
    ensure_file_exists(args.input_path, "Prepared raw CSV")

    source_dataframe = pd.read_csv(args.input_path)
    total_rows = len(source_dataframe)

    it_developers_dataframe = filter_it_developers(source_dataframe)
    it_rows = len(it_developers_dataframe)

    labeled_dataframe = add_level_label(it_developers_dataframe)
    labeled_rows = len(labeled_dataframe)

    print_filtering_summary(
        total_rows=total_rows,
        it_rows=it_rows,
        labeled_rows=labeled_rows,
    )

    plot_class_balance(
        y=labeled_dataframe["dev_level"],
        output_path=args.output_dir / "class_balance.png",
        title="Class balance: junior/middle/senior (IT developers, strict filter)",
    )

    features, target = build_features_and_target(labeled_dataframe, DEFAULT_FEATURES)

    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
        shuffle=True,
    )

    print("=== Data summary ===")
    print(f"Input file: {args.input_path}")
    print(f"Labeled IT developers: {labeled_rows}")
    print("Class distribution:")
    print(target.value_counts())
    print()

    most_common_class = y_train.value_counts().idxmax()
    baseline_predictions = pd.Series(
        [most_common_class] * len(y_valid),
        index=y_valid.index,
    )

    print("=== Baseline (predict most frequent class) ===")
    print(f"Most common class in train: {most_common_class}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_valid, baseline_predictions):.4f}")
    print(classification_report(y_valid, baseline_predictions, digits=4, zero_division=0))

    preprocessor = make_preprocessor(DEFAULT_FEATURES)
    classifier = LinearSVC(
        class_weight="balanced",
        random_state=args.random_state,
    )
    model_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )

    model_pipeline.fit(X_train, y_train)
    predictions = model_pipeline.predict(X_valid)

    print("\n=== Model: LinearSVC ===")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_valid, predictions):.4f}")
    print(classification_report(y_valid, predictions, digits=4, zero_division=0))

    print(f"\nSaved class balance plot to: {args.output_dir / 'class_balance.png'}")


if __name__ == "__main__":
    main()
