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
