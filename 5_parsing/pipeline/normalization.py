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

    def _convert_salary_to_usd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert salary to RUB using fixed exchange rates based on currency column."""
        if "salary" not in df.columns or "currency" not in df.columns:
            return df

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

        rate_series = cur.map(rates)
        rate_series = rate_series.fillna(1.0)
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

        position_norm = self._normalize_position(df["Ищет работу на должность:"])
        last_position_norm = self._normalize_position(df["Последеняя/нынешняя должность"])

        df = df.copy()
        df["salary"] = self._convert_salary_to_usd(df)
        df["position"] = position_norm
        df["last_position"] = last_position_norm
        df["last_work"] = df["Последенее/нынешнее место работы"]

        df = df.drop(
            columns=[
                "Ищет работу на должность:",
                "Последеняя/нынешняя должность",
                "Последенее/нынешнее место работы",
            ],
            errors="ignore",
        )

        context["df"] = df
        print("Done")
        return context
