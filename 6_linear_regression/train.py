import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
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
        help="Path to y_data.npy (target salaries).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="../resources/salary_ridge_model.joblib",
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
        default=1.0,
        help="Ridge regularization strength (alpha).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/valid split.",
    )
    return parser.parse_args()


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
    if y.ndim > 1:
        y = y.ravel()

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
    ) = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    # Стандартизация признаков: Ridge чувствителен к масштабу фич
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    model = Ridge(alpha=args.alpha, random_state=args.random_state)
    model.fit(X_train_scaled, y_train)

    # Оценим качество на валидации
    y_pred_valid = model.predict(X_valid_scaled)
    mae_valid = mean_absolute_error(y_valid, y_pred_valid)
    rmse_valid = mean_squared_error(y_valid, y_pred_valid)
    r2_valid = r2_score(y_valid, y_pred_valid)

    mean_y_valid = y_valid.mean()
    median_y_valid = np.median(y_valid)

    nmae_mean = mae_valid / mean_y_valid if mean_y_valid != 0 else np.nan
    nmae_median = mae_valid / median_y_valid if median_y_valid != 0 else np.nan

    eps = 1e-6
    mape_valid = np.mean(
        np.abs((y_valid - y_pred_valid) / np.maximum(np.abs(y_valid), eps))
    ) * 100

    print("=== Validation metrics ===")
    print(f"MAE:           {mae_valid:.2f} RUB")
    print(f"RMSE:          {rmse_valid:.2f} RUB")
    print(f"R^2:           {r2_valid:.4f}")
    print(f"NMAE (mean y):   {nmae_mean:.3f} (~{nmae_mean * 100:.1f}%)")
    print(f"NMAE (median y): {nmae_median:.3f} (~{nmae_median * 100:.1f}%)")
    print(f"MAPE:            {mape_valid:.1f}%")

    # Сохраняем И скейлер, И модель, чтобы в app.py применять те же преобразования
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "scaler": scaler,
        "model": model,
    }
    joblib.dump(artifact, output_path)

    print(f"Trained Ridge model saved to {output_path}")


if __name__ == "__main__":
    main()
