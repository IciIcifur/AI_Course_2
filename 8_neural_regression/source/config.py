"""Конфигурация обучения модели."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_X_PATH = Path("../data/input/x_data.npy")
DEFAULT_Y_PATH = Path("../data/input/y_data.npy")
DEFAULT_OUTPUT_PATH = Path("../resources/salary_model.pt")

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN_DIMS = (256, 128, 64)
DEFAULT_DROPOUT = 0.3

MLFLOW_TRACKING_URI = "http://kamnsv.com:55000/"
MLFLOW_EXPERIMENT_NAME = "LIne Regression HH"
MLFLOW_MODEL_NAME = "dobrovolskaya_olesya_fcn"


@dataclass(frozen=True)
class TrainConfig:
    """Параметры запуска обучения."""

    x_path: Path
    y_path: Path
    output_path: Path
    test_size: float
    random_state: int
    epochs: int
    batch_size: int
    lr: float
    hidden_dims: tuple[int, ...]
    dropout: float
