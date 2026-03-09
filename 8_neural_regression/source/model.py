"""Архитектура полносвязной нейронной сети для регрессии."""

from __future__ import annotations

import torch
import torch.nn as nn


class SalaryFCN(nn.Module):
    """Полносвязная нейронная сеть для предсказания зарплаты."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        """Инициализирует архитектуру сети.

        Аргументы:
            input_dim: размерность входного вектора признаков.
            hidden_dims: размеры скрытых слоёв.
            dropout: вероятность dropout между слоями.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Выполняет прямой проход.

        Аргументы:
            x: батч входных признаков.

        Вернёт:
            Предсказания в виде одномерного тензора.
        """
        return self.network(x).squeeze(1)