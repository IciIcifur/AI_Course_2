"""Точка входа для обучения модели FCN на данных HH.ru."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import (
    DEFAULT_X_PATH,
    DEFAULT_Y_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_DROPOUT,
    TrainConfig,
)
from data import load_arrays, deduplicate, split
from metrics import compute_metrics, print_metrics
from model import SalaryFCN
from tracking import setup_experiment, log_run
from trainer import fit, predict


def parse_args() -> TrainConfig:
    """Разбирает аргументы командной строки.

    Вернёт:
        Конфигурацию обучения.
    """
    parser = argparse.ArgumentParser(
        description="Обучение FCN-модели для предсказания зарплат HH.ru."
    )
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X_PATH)
    parser.add_argument("--y-path", type=Path, default=DEFAULT_Y_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    args = parser.parse_args()

    return TrainConfig(
        x_path=args.x_path,
        y_path=args.y_path,
        output_path=args.output_path,
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        dropout=args.dropout,
    )


def require_file(path: Path, description: str) -> None:
    """Проверяет существование файла.

    Аргументы:
        path: путь к файлу.
        description: описание файла для сообщения об ошибке.

    Исключения:
        FileNotFoundError: если файл не найден.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{description} не найден: {path}")


def main() -> None:
    """Запускает полный пайплайн обучения."""
    cfg = parse_args()

    require_file(cfg.x_path, "Файл признаков")
    require_file(cfg.y_path, "Файл целевой переменной")

    x, y = load_arrays(cfg.x_path, cfg.y_path)
    x, y = deduplicate(x, y)
    x_train, x_test, y_train, y_test = split(x, y, cfg.test_size, cfg.random_state)

    input_dim = x_train.shape[1]
    model = SalaryFCN(
        input_dim=input_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )

    scaler = fit(model, x_train, y_train, cfg.epochs, cfg.batch_size, cfg.lr)

    y_pred_train = predict(model, scaler, x_train)
    y_pred_test = predict(model, scaler, x_test)

    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)

    print_metrics("train", train_metrics)
    print_metrics("test", test_metrics)

    setup_experiment()
    run_id = log_run(
        model=model,
        params={
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "hidden_dims": str(cfg.hidden_dims),
            "dropout": cfg.dropout,
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
        },
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )
    print(f"MLflow run_id: {run_id}")

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scaler": scaler, "model_state": model.state_dict()}, cfg.output_path)
    print(f"Модель сохранена: {cfg.output_path}")


if __name__ == "__main__":
    main()
