from __future__ import annotations

from typing import Final

import pandas as pd

from .base import Handler


class EncodingHandler(Handler):
    """Perform final feature encoding and cleanup before splitting."""

    DEFAULT_MIN_SALARY: Final[float] = 5_000.0
    DEFAULT_MAX_SALARY: Final[float] = 1_000_000.0

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
        """Drop rows with unknown business_trips or education_level.

        :param df: dataframe with raw/processed features
        :type df: pd.DataFrame
        :return: dataframe without rows containing unknown categories
        :rtype: pd.DataFrame
        """
        if "business_trips" in df.columns:
            df = df[df["business_trips"] != "unknown"]

        if "education_level" in df.columns:
            df = df[df["education_level"] != "unknown"]

        return df

    def _encode_binary_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode simple binary flags (sex, relocation, has_car).

        :param df: dataframe with binary columns
        :type df: pd.DataFrame
        :return: dataframe with encoded binary columns
        :rtype: pd.DataFrame
        """
        df = df.copy()

        if "sex" in df.columns:
            df["sex"] = (df["sex"] == "Мужчина").astype(int)

        if "relocation" in df.columns:
            df["relocation"] = df["relocation"].astype(int)

        if "has_car" in df.columns:
            df["has_car"] = df["has_car"].astype(int)

        return df

    def _fix_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round and fix numeric types for experience and education year.

        :param df: dataframe with numeric columns
        :type df: pd.DataFrame
        :return: dataframe with fixed numeric types
        :rtype: pd.DataFrame
        """
        df = df.copy()

        if "experience_years" in df.columns:
            df["experience_years"] = df["experience_years"].round(2)

        if "education_last_year" in df.columns:
            df["education_last_year"] = df["education_last_year"].astype("Int64")

        return df

    def _keep_top_categories(
            self,
            series: pd.Series,
            top_n: int,
            other: str = "OTHER",
    ) -> pd.Series:
        """Replace rare categories with a shared "OTHER" label.

        :param series: source categorical series
        :type series: pd.Series
        :param top_n: number of most frequent categories to keep
        :type top_n: int
        :param other: value to use for rare categories
        :type other: str
        :return: transformed series with rare categories replaced
        :rtype: pd.Series
        """
        s = series.fillna("UNKNOWN").astype(str)
        top_values = s.value_counts().head(top_n).index
        return s.where(s.isin(top_values), other=other)

    def _one_hot_high_cardinality(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode high-cardinality categoricals safely by keeping top-N.

        :param df: source dataframe
        :type df: pd.DataFrame
        :return: dataframe with one-hot encoded categorical columns
        :rtype: pd.DataFrame
        """
        df = df.copy()

        top_cfg = {
            "city": 500,
            "position": 500,
            "last_position": 500,
            "currency": 8,
        }

        cols = [col for col in top_cfg if col in df.columns]
        if not cols:
            return df

        for col in cols:
            df[col] = self._keep_top_categories(
                df[col],
                top_n=top_cfg[col],
                other="OTHER",
            )

        df = pd.get_dummies(
            df,
            columns=cols,
            prefix=cols,
            drop_first=False,
            dummy_na=False,
        )
        return df

    def _gradational_label_encoding(self, df: pd.DataFrame, map_per_column: dict) -> pd.DataFrame:
        """Label-encode ordinal columns using provided mappings.

        :param df: source dataframe
        :type df: pd.DataFrame
        :param map_per_column: mappings for each column
        :type map_per_column: dict
        :return: dataframe with encoded ordinal columns
        :rtype: pd.DataFrame
        """
        df = df.copy()

        for col, mapping in map_per_column.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).astype("int64")

        return df

    def _encode_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert multi-valued schedule column to multi-hot features.

        :param df: source dataframe
        :type df: pd.DataFrame
        :return: dataframe with schedule multi-hot columns
        :rtype: pd.DataFrame
        """
        if "schedule" not in df.columns:
            return df

        parts = df["schedule"].fillna("").astype(str).str.split("|")
        tokens = sorted({t for row in parts for t in row if t})

        if not tokens:
            return df.drop(columns=["schedule"])

        schedule_df = pd.DataFrame(
            {
                f"schedule__{tok}": parts.apply(lambda row, token=tok: int(token in row))
                for tok in tokens
            },
            index=df.index,
        )

        df_out = df.drop(columns=["schedule"])
        df_out = pd.concat([df_out, schedule_df], axis=1)

        return df_out

    def _drop_raw_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop raw text and unused original columns.

        :param df: source dataframe
        :type df: pd.DataFrame
        :return: dataframe without raw/unneeded columns
        :rtype: pd.DataFrame
        """
        drop_cols = [col for col in df.columns if col.startswith("raw_")]
        drop_cols.extend(
            [
                "last_work",
                "Пол, возраст",
                "ЗП",
                "Город",
                "Опыт (двойное нажатие для полной версии)",
                "Образование и ВУЗ",
            ]
        )
        return df.drop(columns=drop_cols, errors="ignore")

    def _drop_extreme_salaries(
            self,
            df: pd.DataFrame,
            min_salary: float = DEFAULT_MIN_SALARY,
            max_salary: float = DEFAULT_MAX_SALARY,
    ) -> pd.DataFrame:
        """Drop rows with salary outside the [min_salary, max_salary] range.

        :param df: source dataframe
        :type df: pd.DataFrame
        :param min_salary: minimal allowed salary
        :type min_salary: float
        :param max_salary: maximal allowed salary
        :type max_salary: float
        :return: filtered dataframe
        :rtype: pd.DataFrame
        """
        if self.target_column not in df.columns:
            return df

        before = len(df)
        mask = (df[self.target_column] >= min_salary) & (df[self.target_column] <= max_salary)
        df = df[mask]

        dropped = before - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with salary outside [{min_salary}, {max_salary}] RUB.")

        return df

    def _impute_missing_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in numeric columns using median.

        :param df: source dataframe
        :type df: pd.DataFrame
        :return: dataframe with numeric NaNs filled
        :rtype: pd.DataFrame
        """
        df = df.copy()

        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        return df

    def _drop_non_numeric_except_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop remaining non-numeric columns except the target column.

        :param df: source dataframe
        :type df: pd.DataFrame
        :return: dataframe with only numeric features and target
        :rtype: pd.DataFrame
        """
        non_numeric = df.select_dtypes(include=["object", "string"]).columns.tolist()
        non_numeric = [col for col in non_numeric if col != self.target_column]
        return df.drop(columns=non_numeric, errors="ignore")

    def _cast_bools_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast boolean columns to integer type.

        :param df: source dataframe
        :type df: pd.DataFrame
        :return: dataframe with bool columns cast to int
        :rtype: pd.DataFrame
        """
        df = df.copy()

        bool_cols = df.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)

        return df

    def _dedup_by_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate duplicates by feature keys using median salary.

        :param df: source dataframe
        :type df: pd.DataFrame
        :return: aggregated dataframe
        :rtype: pd.DataFrame
        """
        if self.target_column not in df.columns:
            return df

        key_cols = [
            "age",
            "sex",
            "experience_years",
            "education_level",
            "has_master",
            "education_last_year",
            "city",
            "position",
            "last_position",
            "relocation",
            "business_trips",
            "schedule",
            "has_car",
            "currency",
        ]
        key_cols = [col for col in key_cols if col in df.columns]

        if not key_cols:
            return df

        return df.groupby(key_cols, as_index=False)[self.target_column].median()

    def process(self, context: dict) -> dict:
        """Encode categorical features and clean the DataFrame.

        :param context: current pipeline context shared between all handlers
        :type context: dict
        :return: updated context with encoded and cleaned DataFrame
        :rtype: dict
        """
        print("\nENCODING COLUMNS...")

        gradational_columns_map = {
            "education_level": {"school": 0, "vocational": 1, "higher": 2},
            "business_trips": {"none": 0, "rare": 1, "regular": 2},
        }

        df: pd.DataFrame = context["df"].copy()

        df = self._drop_unknown_categories(df)
        df = self._encode_binary_flags(df)
        df = self._fix_numeric_types(df)
        df = self._dedup_by_features(df)

        df = self._gradational_label_encoding(df, gradational_columns_map)
        df = self._one_hot_high_cardinality(df)
        df = self._encode_schedule(df)

        df = self._drop_raw_text_columns(df)
        df = self._impute_missing_numeric(df)
        df = self._drop_non_numeric_except_target(df)
        df = self._drop_extreme_salaries(df)
        df = self._cast_bools_to_int(df)

        context["df"] = df
        print("Done")
        return context
