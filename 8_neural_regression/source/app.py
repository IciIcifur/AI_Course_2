"""Предсказание зарплат с помощью обученной FCN-модели."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_OUTPUT_PATH, DEFAULT_HIDDEN_DIMS, DEFAULT_DROPOUT
from model import SalaryFCN
from trainer import predict


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Вернёт:
        Пространство имён с аргументами.
    """
    parser = argparse.ArgumentParser(
        description="Предсказание зарплат по обученной FCN-модели."
    )
    parser.add_argument("x_path", type=Path, help="Путь к файлу x_data.npy с признаками.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Путь к сохранённому артефакту модели (.pt).",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    """Проверяет существование файла.

    Аргументы:
        path: путь к файлу.
        description: описание для сообщения об ошибке.

    Исключения:
        FileNotFoundError: если файл не найден.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{description} не найден: {path}")


def main() -> None:
    """Загружает модель и выводит предсказания."""
    args = parse_args()

    require_file(args.x_path, "Файл признаков")
    require_file(args.model_path, "Файл модели")

    x = np.load(args.x_path)

    artifact = torch.load(args.model_path, weights_only=False)
    scaler = artifact["scaler"]

    input_dim = x.shape[1]
    model = SalaryFCN(
        input_dim=input_dim,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        dropout=DEFAULT_DROPOUT,
    )
    model.load_state_dict(artifact["model_state"])

    y_pred = predict(model, scaler, x)

    for value in y_pred:
        print(float(value))


if __name__ == "__main__":
    main()
