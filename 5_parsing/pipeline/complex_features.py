import numpy as np
import pandas as pd

from .base import Handler


class ComplexHandler(Handler):
    """Extract education level features from the 'Образование и ВУЗ' column."""

    def __init__(self, next_handler=None, keep_raw: bool = True):
        """Initialize handler with optional raw education preservation.

        :param next_handler: next handler to be executed in the pipeline
        :type next_handler: Handler or None
        :param keep_raw: whether to keep raw education text in a separate column
        :type keep_raw: bool
        """
        super().__init__(next_handler)
        self.keep_raw = keep_raw

    def _parse_education_level(self, series: pd.Series) -> pd.Series:
        """Parse normalized education level from the 'Образование и ВУЗ' column.

        :param series: source column with raw education description
        :type series: pd.Series
        :return: series with normalized education level codes
        :rtype: pd.Series
        """
        s = series.astype(str).str.lower()

        conditions = [
            s.str.contains("магистр", na=False),
            s.str.contains("бакалавр", na=False),
            s.str.contains("высшее", na=False),
            s.str.contains("среднее профессиональное", na=False)
            | s.str.contains("среднее специальное", na=False),
            s.str.contains("среднее общее", na=False),
        ]
        choices = [
            "master",
            "bachelor",
            "higher",
            "vocational",
            "school",
        ]
        level = pd.Series(
            np.select(conditions, choices, default="unknown"),
            index=series.index,
            dtype="string",
        )
        return level

    def process(self, context: dict) -> dict:
        """Add education level features to the DataFrame.

        :param context: current pipeline context shared between all handlers
        :type context: dict
        :return: updated context with education features
        :rtype: dict
        """
        print("\nCOMPLEX FEATURES...")

        df: pd.DataFrame = context["df"]

        edu_level = self._parse_education_level(df["Образование и ВУЗ"])
        df = df.copy()
        df["education_level"] = edu_level

        if self.keep_raw:
            df["raw_education"] = df["Образование и ВУЗ"]

        context["df"] = df
        print("Done")
        return context
