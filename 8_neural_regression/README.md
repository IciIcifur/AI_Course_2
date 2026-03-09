# HH.ru Neural Regression

Обучение полносвязной нейронной сети (FCN) для предсказания желаемой зарплаты
по данным резюме с HH.ru с трекингом экспериментов в MLflow.

## Запуск обучения

```bash
python 8_neural_regression/source/train.py --x-path data/input/x_data.npy --y-path data/input/y_data.npy --epochs 300
```

## Запуск предсказания

```bash
python 8_neural_regression/source/app.py data/input/x_data.npy --model-path resources/salary_model.pt
```

## Создание виртуального окружения

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Структура проекта

```
8_neural_regression/
  source/
    config.py     - константы и датакласс конфигурации
    data.py       - загрузка, дедупликация, разбивка данных
    model.py      - архитектура FCN
    trainer.py    - цикл обучения и предсказание
    metrics.py    - подсчёт метрик качества
    tracking.py   - интеграция с MLflow
    train.py      - точка входа обучения
    app.py        - точка входа инференса
```

## Архитектура модели

Полносвязная нейронная сеть (FCN): Linear → BatchNorm → ReLU → Dropout,
три скрытых слоя размерностью 512 → 256 → 128. Целевая переменная
логарифмируется перед обучением (log1p), предсказания переводятся обратно
через expm1.

## Трекинг в MLflow

- Сервер: http://kamnsv.com:55000/
- Эксперимент: `LIne Regression HH`
- Название модели: `dobrovolskaya_olesya_fcn`
- Метрика: `r2_score_test`

## Метрики лучшего запуска

| Метрика | Train      | Test       |
|---------|------------|------------|
| R²      | 0.689      | 0.516      |
| MAE     | 18 444 RUB | 26 500 RUB |
| RMSE    | 36 968 RUB | 46 830 RUB |
| MAPE    | 20.6%      | 34.8%      |

## Зависимости

Управляются через `pyproject.toml` в корне репозитория.