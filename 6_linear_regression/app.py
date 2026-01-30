import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np

DEFAULT_MODEL_PATH = (
        Path(__file__).resolve().parents[1] / "resources" / "salary_model.joblib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict desired salaries using a trained Ridge regression model."
    )
    parser.add_argument(
        "x_path",
        type=str,
        help="Path to x_data.npy with features.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained model artifact (.joblib).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x_path = Path(args.x_path)
    if not x_path.is_file():
        raise FileNotFoundError(f"x_data file not found: {x_path}")

    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    X = np.load(x_path)

    artifact = joblib.load(model_path)
    scaler = artifact["scaler"]
    model = artifact["model"]

    X_scaled = scaler.transform(X)

    y_pred_log = model.predict(X_scaled)

    y_pred = np.expm1(y_pred_log)

    salaries: List[float] = [float(v) for v in y_pred.tolist()]
    for salary in salaries:
        print(salary)


if __name__ == "__main__":
    main()
