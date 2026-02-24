# HH.ru linear regression model

Обучение модели линейной регрессии (RidgeCV) на датасете HH.ru, полученном на этапе
парсинга (`x_data.npy`, `y_data.npy`).

## Обучение

Скрипт обучения: `source/train.py`.

По умолчанию он ожидает данные здесь (пути относительно `source/`):

- `../data/input/x_data.npy`
- `../data/input/y_data.npy`

И сохраняет артефакт модели сюда:

- `../resources/salary_model.joblib`

Запуск с дефолтами:

```bash
python source/train.py
```

Или с явными путями:

```bash
python source/train.py --x-path path/to/x_data.npy --y-path path/to/y_data.npy --output-path resources/salary_model.joblib
```

Артефакт `salary_model.joblib` содержит и скейлер, и модель (joblib dict с ключами `scaler` и `model`).

## Предсказания

Скрипт предсказаний: `source/app.py`.

```bash
python source/app.py path/to/x_data.npy --model-path resources/salary_model.joblib
```

Важно: модель обучается на `log1p(salary)`, поэтому при предсказании значение переводится обратно через `expm1()`.

## Стратегия обучения модели

1. Удаляем полные дубликаты строк (дубликаты только по X должны быть обработаны ранее при парсинге).
2. Делим данные на train/validation через `train_test_split`.
3. Масштабируем признаки `StandardScaler`.
4. Обучаем `RidgeCV` с подбором коэффициента регуляризации `alpha` на лог-трансформированном таргете `log1p(y)`.
5. Считаем метрики на валидации в исходной шкале зарплат (RUB): MAE, RMSE, R², MAPE и т.д.
6. Сохраняем веса модели вместе со скейлером в `salary_model.joblib`.

Метрики обученной модели, веса которой сохранены в репозитории (`alpha` = 2222.99):

```
=== Validation metrics (Ridge) ===
MAE:             28219.91 RUB
RMSE:            48683.85 RUB
R^2:             0.4725
NMAE (mean y):   0.349 (~34.9%)
NMAE (median y): 0.470 (~47.0%)
MAPE:            37.7%
```