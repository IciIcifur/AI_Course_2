"""Загрузка, очистка и разделение данных."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def load_arrays(x_path, y_path) -> tuple[np.ndarray, np.ndarray]:
    """Загружает массивы признаков и целевой переменной из файлов.

    Аргументы:
        x_path: путь к файлу с признаками.
        y_path: путь к файлу с целевой переменной.

    Вернёт:
        Пару массивов (X, y).
    """
    x = np.load(x_path)
    y = np.load(y_path).astype(float)
    return x, y


def deduplicate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Удаляет полностью дублирующиеся строки по паре (X, y).

    Аргументы:
        x: матрица признаков.
        y: вектор целевой переменной.

    Вернёт:
        Очищенную пару (X, y).
    """
    if y.ndim != 1:
        y = y.ravel()
    data = np.concatenate([x, y.reshape(-1, 1)], axis=1)
    unique = np.unique(data, axis=0)
    return unique[:, :-1], unique[:, -1]


def split(
        x: np.ndarray,
        y: np.ndarray,
        test_size: float,
        random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Разделяет данные на обучающую и тестовую выборки.

    Аргументы:
        x: матрица признаков.
        y: вектор целевой переменной.
        test_size: доля тестовой выборки.
        random_state: seed для воспроизводимости.

    Вернёт:
        Кортеж (x_train, x_test, y_train, y_test).
    """
    return train_test_split(x, y, test_size=test_size, random_state=random_state, shuffle=True)
