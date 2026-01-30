import pandas as pd

from .base import Handler


class EncodingHandler(Handler):
    """Perform final feature encoding and cleanup before splitting."""

    def __init__(self, target_column: str = "salary", next_handler=None):
        """Initialize encoder with the name of the target column.

        :param target_column: name of the target column to keep unchanged
        :type target_column: str
        :param next_handler: next handler to be executed in the pipeline
        :type next_handler: Handler or None
        """
        super().__init__(next_handler)
        self.target_column = target_column

    def _drop_unknown_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with unknown business_trips or education_level."""
        if "business_trips" in df.columns:
            df = df[df["business_trips"] != "unknown"]

        if "education_level" in df.columns:
            df = df[df["education_level"] != "unknown"]

        return df

    def _encode_binary_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode simple binary flags (sex, relocation, has_car)."""
        if "sex" in df.columns:
            df["sex"] = (df["sex"] == "Мужчина").astype(int)

        if "relocation" in df.columns:
            df["relocation"] = df["relocation"].astype(int)

        if "has_car" in df.columns:
            df["has_car"] = df["has_car"].astype(int)

        return df

    def _fix_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round and fix numeric types for experience and education year."""
        if "experience_years" in df.columns:
            df["experience_years"] = df["experience_years"].round(2)

        if "education_last_year" in df.columns:
            df["education_last_year"] = df["education_last_year"].astype("Int64")

        return df

    def _label_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode given column in int64"""
        cat_cols: list[str] = ['currency', 'city', 'position', 'last_position']
        for column in cat_cols:
            if column in df.columns:
                series = df[column].fillna("").astype(str)
                unique = sorted(series.unique())
                mapping = {value: idx for idx, value in enumerate(unique)}
                df[column] = series.map(mapping).astype("int64")
        return df

    def _gradational_label_encoding(self, df: pd.DataFrame, map_per_column: dict):
        """Label-encode given column with given mappings in int64"""
        for column in map_per_column.keys():
            if column in df.columns:
                df[column] = df[column].map(map_per_column[column]).astype("int64")
        return df

    def _encode_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert multi-valued schedule column to multi-hot features."""
        if "schedule" in df.columns:
            parts = df["schedule"].fillna("").astype(str).str.split("|")
            tokens = sorted({t for row in parts for t in row if t})
            for t in tokens:
                if not t:
                    continue
                col_name = f"schedule__{t}"
                df[col_name] = parts.apply(lambda row, tok=t: int(tok in row))
            df = df.drop(columns=["schedule"])
        return df

    def _drop_raw_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop raw text and unused original columns."""
        drop_cols: list[str] = []
        for col in df.columns:
            if col.startswith("raw_"):
                drop_cols.append(col)
        drop_cols.extend(["last_work", "Пол, возраст", "ЗП", "Город", "Опыт (двойное нажатие для полной версии)",
                          "Образование и ВУЗ", ])
        df = df.drop(columns=drop_cols, errors="ignore")
        return df

    def _drop_extreme_salaries(self, df: pd.DataFrame, min_salary: float = 5_000.0,
                               max_salary: float = 1_000_000.0, ) -> pd.DataFrame:
        """Drops rows with salary outside [min_salary, max_salary] range."""
        if "salary" not in df.columns:
            return df

        before = len(df)
        mask = (df["salary"] >= min_salary) & (df["salary"] <= max_salary)
        df = df[mask]

        after = len(df)
        dropped = before - after
        if dropped > 0:
            print(
                f"Dropped {dropped} rows with salary outside [{min_salary}, {max_salary}] RUB."
            )

        return df

    def _impute_missing_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in numeric columns using median."""
        for col in ["age", "experience_years", "education_last_year"]:
            if col in df.columns:
                median = df[col].median()
                df[col] = df[col].fillna(median)

        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(median)

        return df

    def _drop_non_numeric_except_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop remaining non-numeric columns except the target column."""
        non_numeric = df.select_dtypes(include=["object", "string"]).columns.tolist()
        non_numeric = [c for c in non_numeric if c != self.target_column]
        df = df.drop(columns=non_numeric, errors="ignore")
        return df

    def _cast_bools_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast boolean columns to integer type."""
        bool_cols = df.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)
        return df

    def process(self, context: dict) -> dict:
        """Encode categorical features and clean the DataFrame.

        :param context: current pipeline context shared between all handlers
        :type context: dict
        :return: updated context with encoded and cleaned DataFrame
        :rtype: dict
        """
        print("\nENCODING COLUMNS...")

        gradational_columns_map = {
            'education_level':  {'school': 0,  'vocational': 1,  'higher': 2},
            'business_trips': {'none': 0,  'rare': 1, 'regular': 2}
        }

        df: pd.DataFrame = context["df"].copy()

        df = self._drop_unknown_categories(df)
        df = self._encode_binary_flags(df)
        df = self._fix_numeric_types(df)

        df = self._gradational_label_encoding(df, gradational_columns_map)
        df = self._label_encode_categoricals(df)

        df = self._encode_schedule(df)
        df = self._drop_raw_text_columns(df)
        df = self._impute_missing_numeric(df)
        df = self._drop_non_numeric_except_target(df)
        df = self._drop_extreme_salaries(df)
        df = self._cast_bools_to_int(df)

        context["df"] = df
        print("Done")
        return context
