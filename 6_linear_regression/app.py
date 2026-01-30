"""Predict salaries using a trained RidgeCV artifact (scaler + model).

The model is trained on log1p(salary), so predictions are converted back with
expm1() before printing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "resources" / "salary_model.joblib"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Predict desired salaries using a trained RidgeCV model."
    )
    parser.add_argument(
        "x_path",
        type=Path,
        help="Path to x_data.npy with features.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model artifact (.joblib).",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    """Raise FileNotFoundError if `path` does not exist."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def main() -> None:
    """Entrypoint."""
    args = parse_args()

    require_file(args.x_path, "x_data file")
    require_file(args.model_path, "Model file")

    x = np.load(args.x_path)

    artifact: dict[str, Any] = joblib.load(args.model_path)
    scaler = artifact["scaler"]
    model = artifact["model"]

    x_scaled = scaler.transform(x)

    # model predicts log1p(salary)
    y_pred_log = model.predict(x_scaled)
    y_pred = np.expm1(y_pred_log)

    for value in y_pred:
        print(float(value))


if __name__ == "__main__":
    main()
