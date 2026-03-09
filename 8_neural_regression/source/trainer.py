"""Обучение и оценка модели."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import SalaryFCN


def make_loader(
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
) -> DataLoader:
    """Создаёт DataLoader из numpy-массивов.

    Аргументы:
        x: матрица признаков.
        y: вектор целевой переменной.
        batch_size: размер батча.
        shuffle: перемешивать ли данные.

    Вернёт:
        DataLoader для итерации по батчам.
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def fit(
        model: SalaryFCN,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
) -> StandardScaler:
    """Обучает модель на тренировочных данных.

    Аргументы:
        model: экземпляр нейронной сети.
        x_train: тренировочные признаки.
        y_train: тренировочная целевая переменная.
        epochs: количество эпох обучения.
        batch_size: размер батча.
        lr: скорость обучения.

    Вернёт:
        Обученный StandardScaler для трансформации признаков.
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    y_log = np.log1p(y_train)

    loader = make_loader(x_scaled, y_log, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch + 1}/{epochs} — loss: {total_loss / len(loader):.4f}")

    return scaler


def predict(
        model: SalaryFCN,
        scaler: StandardScaler,
        x: np.ndarray,
) -> np.ndarray:
    """Выполняет предсказание на новых данных.

    Аргументы:
        model: обученная нейронная сеть.
        scaler: обученный StandardScaler.
        x: матрица признаков.

    Вернёт:
        Предсказания в исходном масштабе (рубли).
    """
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_log = model(x_tensor).numpy()

    return np.expm1(y_log)
