"""Train PoC classifier for developer level (junior/middle/senior) from HH.ru resumes.

PoC notes:
- IT developers are selected by heuristic rules (see labelling.py).
- Target y (dev_level) is formed heuristically from title markers and/or experience.
- Model quality is evaluated on a hold-out split with classification_report.

This script keeps a single best-performing model (LinearSVC) for simplicity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from features import DEFAULT_FEATURES, build_xy, make_preprocessor
from labelling import add_level_label, filter_it
from plots import plot_class_balance


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Train PoC classifier for junior/middle/senior developer level."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/hh_prepared_raw.csv"),
        help="Path to prepared raw CSV (from 5_parsing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resources"),
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use as validation set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    """Raise FileNotFoundError if `path` does not exist."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def main() -> None:
    """Entrypoint."""
    args = parse_args()
    require_file(args.input_path, "Prepared raw CSV")

    df = pd.read_csv(args.input_path)
    n0 = len(df)

    df_it = filter_it(df)
    n1 = len(df_it)

    df_labeled = add_level_label(df_it)
    n2 = len(df_labeled)

    print("=== Filtering summary ===")
    print(f"Total rows in prepared_raw: {n0}")
    print(f"After IT developer filter:  {n1} (dropped {n0 - n1}, {(n0 - n1) / max(n0, 1) * 100:.1f}%)")
    print(f"After level labeling:       {n2} (dropped {n1 - n2}, {(n1 - n2) / max(n1, 1) * 100:.1f}%)")
    print()

    plot_class_balance(
        df_labeled["dev_level"],
        output_path=args.output_dir / "class_balance.png",
        title="Class balance: junior/middle/senior in IT",
    )

    X, y = build_xy(df_labeled, DEFAULT_FEATURES)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
        shuffle=True,
    )

    print("=== Data summary ===")
    print(f"Input file: {args.input_path}")
    print(f"Labeled IT developers: {len(df_labeled)}")
    print()

    print("Class distribution:")
    print(y.value_counts())
    print()

    preprocessor = make_preprocessor(DEFAULT_FEATURES)

    model = LinearSVC(
        class_weight="balanced",
        random_state=args.random_state,
    )
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)

    print("\n=== Model: LinearSVC ===")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_valid, y_pred):.4f}")
    print(classification_report(y_valid, y_pred, digits=4, zero_division=0))

    print(f"\nSaved class balance plot to: {args.output_dir / 'class_balance.png'}")


if __name__ == "__main__":
    main()
