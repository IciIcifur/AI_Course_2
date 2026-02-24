from __future__ import annotations

import pandas as pd

from .base import Handler


class NormalizeHandler(Handler):
    """Normalize current and last position columns."""

    def _normalize_position(self, series: pd.Series) -> pd.Series:
        """Normalize raw position text to a unified lowercase representation.

        :param series: source column with raw position text
        :type series: pd.Series
        :return: series with normalized position strings
        :rtype: pd.Series
        """
        s = series.fillna("").astype(str).str.strip().str.lower()
        s = s.str.replace(r"\s*,\s*", ", ", regex=True)
        s = s.str.replace(r"\s*/\s*", " / ", regex=True)
        return s

    def _normalize_city(self, series: pd.Series) -> pd.Series:
        """Normalize city names: lowercase, trim, drop text in parentheses.

        :param series: source city column
        :type series: pd.Series
        :return: normalized city names
        :rtype: pd.Series
        """
        s = series.fillna("").astype(str).str.strip().str.lower()

        mask_moscow = s.str.contains(r"\bмосква\b", regex=True)
        s.loc[mask_moscow] = "москва"

        mask_mo = s.str.contains(r"московск(?:ая|ой)\s+обл", regex=True) | s.str.contains(
            r"московская область",
            regex=True,
        )
        s.loc[mask_mo & ~mask_moscow] = "московская область"

        s = s.str.replace(r"\s*\(.*\)$", "", regex=True)
        s = s.str.replace(r"\s+", " ", regex=True)

        return s

    def _convert_salary_to_rub(self, df: pd.DataFrame) -> pd.Series:
        """Convert salary to RUB using fixed exchange rates based on currency column.

        :param df: source dataframe containing 'salary' and 'currency' columns
        :type df: pd.DataFrame
        :return: converted salary series in RUB
        :rtype: pd.Series
        """
        if "salary" not in df.columns or "currency" not in df.columns:
            return df.get("salary")

        rates = {
            "usd": 76.55,
            "eur": 91.46,
            "руб": 1.0,
            "грн": 1.8,
            "azn": 45.03,
            "kzt": 0.15,
            "kgs": 0.88,
            "сум": 0.006,
        }

        salary = df["salary"].astype(float)
        cur = df["currency"].fillna("").astype(str).str.lower()

        rate_series = cur.map(rates).fillna(1.0)
        return (salary * rate_series).astype(float)

    def process(self, context: dict) -> dict:
        """Add normalized position features and drop raw position columns.

        :param context: current pipeline context shared between all handlers
        :type context: dict
        :return: updated context with normalized position features
        :rtype: dict
        """
        print("\nNORMALIZING...")

        df: pd.DataFrame = context["df"]

        df_out = df.copy()
        df_out["salary"] = self._convert_salary_to_rub(df_out)
        df_out["city"] = self._normalize_city(df_out["city"])
        df_out["position"] = self._normalize_position(df_out["Ищет работу на должность:"])
        df_out["last_position"] = self._normalize_position(df_out["Последеняя/нынешняя должность"])
        df_out["last_work"] = df_out["Последенее/нынешнее место работы"]

        df_out = df_out.drop(
            columns=[
                "Ищет работу на должность:",
                "Последеняя/нынешняя должность",
                "Последенее/нынешнее место работы",
            ],
            errors="ignore",
        )

        context["df"] = df_out
        print("Done")
        return context
