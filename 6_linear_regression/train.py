import argparse
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Ridge regression model on HH.ru parsed features."
    )
    parser.add_argument(
        "--x-path",
        type=str,
        default="../data/input/x_data.npy",
        help="Path to x_data.npy (features).",
    )
    parser.add_argument(
        "--y-path",
        type=str,
        default="../data/input/y_data.npy",
        help="Path to y_data.npy (target salaries in RUB).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="../resources/salary_model.joblib",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use as validation set.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="Ridge regularization strength (alpha).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/valid split.",
    )
    return parser.parse_args()


def print_metrics(
        name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    mean_y = y_true.mean()
    median_y = np.median(y_true)
    nmae_mean = mae / mean_y if mean_y != 0 else np.nan
    nmae_median = mae / median_y if median_y != 0 else np.nan

    eps = 1e-6
    mape = (np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)

    print(f"=== Validation metrics ({name}) ===")
    print(f"MAE:             {mae:.2f} RUB")
    print(f"RMSE:            {rmse:.2f} RUB")
    print(f"R^2:             {r2:.4f}")
    print(f"NMAE (mean y):   {nmae_mean:.3f} (~{nmae_mean * 100:.1f}%)")
    print(f"NMAE (median y): {nmae_median:.3f} (~{nmae_median * 100:.1f}%)")
    print(f"MAPE:            {mape:.1f}%")
    print()


def main() -> None:
    args = parse_args()

    x_path = Path(args.x_path)
    y_path = Path(args.y_path)
    output_path = Path(args.output_path)

    if not x_path.is_file():
        raise FileNotFoundError(f"x_data file not found: {x_path}")
    if not y_path.is_file():
        raise FileNotFoundError(f"y_data file not found: {y_path}")

    X = np.load(x_path)
    y = np.load(y_path).astype(float)

    data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    data_unique = np.unique(data, axis=0)
    X = data_unique[:, :-1]
    y = data_unique[:, -1]

    if y.ndim > 1:
        y = y.ravel()

    (X_train, X_valid, y_train, y_valid,) = train_test_split(X, y, test_size=args.test_size,
                                                             random_state=args.random_state, shuffle=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    y_train_log = np.log1p(y_train)

    ridge = RidgeCV(alphas=np.logspace(-4, 6, 50), cv=5)
    ridge.fit(X_train_scaled, y_train_log)

    y_pred = np.expm1(ridge.predict(X_valid_scaled))

    print_metrics('Ridge', y_valid, y_pred)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "scaler": scaler,
        "model": ridge,
    }
    joblib.dump(artifact, output_path)

    print(f"Trained model saved to {output_path}")


if __name__ == "__main__":
    main()
