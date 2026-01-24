import pandas as pd

from .base import Handler


class EncodingHandler(Handler):
    """
    Финальное кодирование признаков перед SplitTargetHandler:
    - sex, relocation, has_car -> 0/1
    - experience_years -> округление до 2 знаков
    - education_last_year -> Int64
    - currency -> one-hot
    - business_trips -> one-hot
    - schedule -> multi-hot (schedule__fullday, ...)
    - education_level -> one-hot
    - position -> label encoding (position_le)
    - дропает сырые текстовые колонки и любые оставшиеся object/string
      (кроме целевого столбца, если он строковый)
    """

    def __init__(self, target_column: str = "salary", next_handler=None):
        super().__init__(next_handler)
        self.target_column = target_column

    def process(self, context: dict) -> dict:
        print("\nENCODING COLUMNS...")

        df: pd.DataFrame = context["df"].copy()

        # до любых преобразований
        if "business_trips" in df.columns:
            before = len(df)
            df = df[df["business_trips"] != "unknown"]
            print(f"Dropped {before - len(df)} rows with business_trips == 'unknown'")

        if "education_level" in df.columns:
            before = len(df)
            df = df[df["education_level"] != "unknown"]
            print(f"Dropped {before - len(df)} rows with education_level == 'unknown'")

        # --- 1. sex / relocation / has_car -> 0/1 ---

        if "sex" in df.columns:
            # Мужчина -> 1, остальные -> 0
            df["sex"] = (df["sex"] == "Мужчина").astype(int)

        if "relocation" in df.columns:
            df["relocation"] = df["relocation"].astype(int)

        if "has_car" in df.columns:
            df["has_car"] = df["has_car"].astype(int)

        # --- 2. experience_years -> округление, education_last_year -> Int64 ---

        if "experience_years" in df.columns:
            df["experience_years"] = df["experience_years"].round(2)

        if "education_last_year" in df.columns:
            df["education_last_year"] = df["education_last_year"].astype("Int64")

        # --- 3. label encoding для position ---

        if "position" in df.columns:
            pos_series = df["position"].fillna("").astype(str)
            uniques = sorted(pos_series.unique())
            mapping = {val: idx for idx, val in enumerate(uniques)}
            df["position"] = pos_series.map(mapping).astype("int64")

        if "last_position" in df.columns:
            pos_series = df["last_position"].fillna("").astype(str)
            uniques = sorted(pos_series.unique())
            mapping = {val: idx for idx, val in enumerate(uniques)}
            df["last_position"] = pos_series.map(mapping).astype("int64")

        # --- 4. multi-hot для schedule ('fullday|remote|...') ---

        if "schedule" in df.columns:
            parts = df["schedule"].fillna("").astype(str).str.split("|")
            tokens = sorted({t for row in parts for t in row if t})
            for t in tokens:
                if not t:
                    continue
                col_name = f"schedule__{t}"
                df[col_name] = parts.apply(lambda row, tok=t: int(tok in row))
            df = df.drop(columns=["schedule"])

        # --- 5. one-hot для "маленьких" категорий ---

        # Явно зафиксируем, что хотим one-hot именно по этим колонкам
        cat_cols = []
        for col in ["currency", "business_trips", "education_level"]:
            if col in df.columns and df[col].dtype.name in ("object", "string"):
                cat_cols.append(col)

        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dummy_na=False)

        # --- 6. выбрасываем очевидные сырьевые текстовые поля ---

        drop_cols = []
        for col in df.columns:
            if col.startswith("raw_"):
                drop_cols.append(col)
        drop_cols.extend(
            [
                "last_work",
                # исходные текстовые колонки из исходного CSV, если ещё остались
                "Пол, возраст",
                "ЗП",
                "Город",
                "Опыт (двойное нажатие для полной версии)",
                "Образование и ВУЗ",
            ]
        )
        df = df.drop(columns=drop_cols, errors="ignore")

        # --- 7. Импутация пропусков в числовых колонках ---

        # Простая стратегия:
        # - age, experience_years, education_last_year -> медиана
        # (можно расширить список при желании)
        for col in ["age", "experience_years", "education_last_year"]:
            if col in df.columns:
                median = df[col].median()
                df[col] = df[col].fillna(median)

        # Если остались какие-то другие числовые колонки с NaN —
        # можно заполнить их нулями или тоже по медиане.
        # Например, по всем numeric-колонкам:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(median)

        # --- 8. удаляем оставшиеся object/string колонки, кроме target_column ---

        # иногда могут остаться неожиданные текстовые признаки;
        # в X мы хотим только числа
        non_numeric = df.select_dtypes(include=["object", "string"]).columns.tolist()
        non_numeric = [c for c in non_numeric if c != self.target_column]
        df = df.drop(columns=non_numeric, errors="ignore")

        # --- 9. приведение bool -> int (если остались) ---

        bool_cols = df.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)

        context["df"] = df
        print("Remaining columns:", df.columns.tolist())
        print("Done")
        return context
