"""Вычисление метрик качества регрессии."""

from __future__ import annotations

from math import sqrt

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EPS = 1e-6


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Вычисляет основные метрики регрессии в исходном масштабе целевой переменной.

    Аргументы:
        y_true: истинные значения целевой переменной.
        y_pred: предсказанные значения.

    Вернёт:
        Словарь с метриками: mae, rmse, r2, mape, nmae_mean, nmae_median.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mean_y = float(np.mean(y_true))
    median_y = float(np.median(y_true))

    nmae_mean = mae / mean_y if mean_y else float("nan")
    nmae_median = mae / median_y if median_y else float("nan")

    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), EPS))) * 100.0
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "nmae_mean": nmae_mean,
        "nmae_median": nmae_median,
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    """Выводит метрики в читаемом формате.

    Аргументы:
        name: название выборки для заголовка.
        metrics: словарь с метриками от compute_metrics.
    """
    print(f"=== Метрики ({name}) ===")
    print(f"MAE:             {metrics['mae']:.2f} RUB")
    print(f"RMSE:            {metrics['rmse']:.2f} RUB")
    print(f"R^2:             {metrics['r2']:.4f}")
    print(f"NMAE (mean y):   {metrics['nmae_mean']:.3f} (~{metrics['nmae_mean'] * 100:.1f}%)")
    print(f"NMAE (median y): {metrics['nmae_median']:.3f} (~{metrics['nmae_median'] * 100:.1f}%)")
    print(f"MAPE:            {metrics['mape']:.1f}%")
    print()
