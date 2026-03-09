"""Трекинг экспериментов в MLflow."""

from __future__ import annotations

import mlflow
import mlflow.pytorch

from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME
from model import SalaryFCN


def setup_experiment() -> None:
    """Настраивает подключение к MLflow и устанавливает эксперимент."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_run(
        model: SalaryFCN,
        params: dict[str, object],
        train_metrics: dict[str, float],
        test_metrics: dict[str, float],
) -> str:
    """Логирует параметры, метрики и модель в MLflow.

    Аргументы:
        model: обученная нейронная сеть.
        params: гиперпараметры обучения.
        train_metrics: метрики на тренировочной выборке.
        test_metrics: метрики на тестовой выборке.

    Вернёт:
        Идентификатор запуска run_id.
    """
    with mlflow.start_run(run_name=MLFLOW_MODEL_NAME) as run:
        mlflow.log_params(params)

        for key, value in train_metrics.items():
            mlflow.log_metric(f"{key}_train", value)

        mlflow.log_metric("r2_score_test", test_metrics["r2"])
        for key, value in test_metrics.items():
            if key != "r2":
                mlflow.log_metric(f"{key}_test", value)

        mlflow.pytorch.log_model(model, name=MLFLOW_MODEL_NAME)

        return run.info.run_id
