"""Train a RidgeCV regression model to predict desired salary from HH.ru features.

The script loads precomputed numpy arrays (x_data.npy and y_data.npy), splits the
data into train/validation parts, standardizes features, trains RidgeCV on the
log-transformed target (log1p), evaluates metrics in the original salary scale,
and saves the trained artifact (scaler + model) to a joblib file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_X_PATH = Path("../data/input/x_data.npy")
DEFAULT_Y_PATH = Path("../data/input/y_data.npy")
DEFAULT_OUTPUT_PATH = Path("../resources/salary_model.joblib")

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

EPS = 1e-6


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training and validation."""

    x_path: Path
    y_path: Path
    output_path: Path
    test_size: float
    random_state: int


def parse_args() -> TrainConfig:
    """Parse CLI arguments into TrainConfig."""
    parser = argparse.ArgumentParser(
        description="Train RidgeCV regression model on HH.ru parsed features."
    )
    parser.add_argument(
        "--x-path",
        type=Path,
        default=DEFAULT_X_PATH,
        help="Path to x_data.npy (features).",
    )
    parser.add_argument(
        "--y-path",
        type=Path,
        default=DEFAULT_Y_PATH,
        help="Path to y_data.npy (target salaries in RUB).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the trained model artifact (.joblib).",
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
        help="Random seed for train/valid split.",
    )
    args = parser.parse_args()

    return TrainConfig(
        x_path=args.x_path,
        y_path=args.y_path,
        output_path=args.output_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )


def require_file(path: Path, description: str) -> None:
    """Raise FileNotFoundError if `path` does not exist."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def deduplicate_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove fully duplicated (X, y) rows.

    Note: this removes only exact duplicates. Duplicates by X-only should be
    handled earlier in the parsing pipeline.
    """
    if y.ndim != 1:
        y = y.ravel()

    data = np.concatenate([x, y.reshape(-1, 1)], axis=1)
    unique = np.unique(data, axis=0)

    x_unique = unique[:, :-1]
    y_unique = unique[:, -1]
    return x_unique, y_unique


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print common regression metrics in the original target scale."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mean_y = float(np.mean(y_true))
    median_y = float(np.median(y_true))

    nmae_mean = mae / mean_y if mean_y else np.nan
    nmae_median = mae / median_y if median_y else np.nan

    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), EPS))) * 100.0
    )

    print(f"=== Validation metrics ({name}) ===")
    print(f"MAE:             {mae:.2f} RUB")
    print(f"RMSE:            {rmse:.2f} RUB")
    print(f"R^2:             {r2:.4f}")
    print(f"NMAE (mean y):   {nmae_mean:.3f} (~{nmae_mean * 100:.1f}%)")
    print(f"NMAE (median y): {nmae_median:.3f} (~{nmae_median * 100:.1f}%)")
    print(f"MAPE:            {mape:.1f}%")
    print()


def train_model(
        x_train: np.ndarray,
        y_train: np.ndarray,
        alphas: np.ndarray,
        cv: int,
) -> tuple[StandardScaler, RidgeCV]:
    """Fit StandardScaler + RidgeCV on log1p(y)."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    y_train_log = np.log1p(y_train)

    model = RidgeCV(alphas=alphas, cv=cv)
    model.fit(x_train_scaled, y_train_log)

    return scaler, model


def main() -> None:
    """Entrypoint."""
    cfg = parse_args()

    require_file(cfg.x_path, "x_data file")
    require_file(cfg.y_path, "y_data file")

    x = np.load(cfg.x_path)
    y = np.load(cfg.y_path).astype(float)

    x, y = deduplicate_xy(x, y)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        shuffle=True,
    )

    alphas = np.logspace(-4, 6, 50)
    scaler, model = train_model(x_train, y_train, alphas=alphas, cv=5)

    x_valid_scaled = scaler.transform(x_valid)
    y_pred_log = model.predict(x_valid_scaled)
    y_pred = np.expm1(y_pred_log)

    print(f"Best alpha: {model.alpha_}")
    print_metrics("RidgeCV", y_valid, y_pred)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact: dict[str, Any] = {
        "scaler": scaler,
        "model": model,
    }
    joblib.dump(artifact, cfg.output_path)

    print(f"Trained model saved to {cfg.output_path}")


if __name__ == "__main__":
    main()
