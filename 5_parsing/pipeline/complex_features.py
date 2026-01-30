import numpy as np
import pandas as pd

from .base import Handler


class ComplexHandler(Handler):
    """Extract education level features from the 'Образование и ВУЗ' column."""

    def __init__(self, next_handler=None, keep_raw: bool = True):
        super().__init__(next_handler)
        self.keep_raw = keep_raw

    def _parse_education_level(self, series: pd.Series) -> pd.DataFrame:
        """Parse normalized education level from the 'Образование и ВУЗ' column.

        :param series: source column with raw education description
        :type series: pd.Series
        :return: dataframe with normalized education level codes and has_master boolean column
        :rtype: pd.Dataframe
        """
        s = series.astype(str).str.lower()

        has_master = s.str.contains("магистр", na=False).astype(int)

        conditions = [
            # высшее
            s.str.contains("высшее", na=False)
            | s.str.contains("бакалавр", na=False)
            | s.str.contains("магистр", na=False),

            # среднее профессиональное
            s.str.contains("среднее профессиональное", na=False)
            | s.str.contains("среднее специальное", na=False),

            # среднее общее
            s.str.contains("среднее общее", na=False),
        ]

        choices = ["higher", "vocational", "school"]

        education_level = pd.Series(
            np.select(conditions, choices, default="school"),
            index=series.index,
            dtype="string",
        )

        return pd.DataFrame(
            {
                "education_level": education_level,
                "has_master": has_master,
            },
            index=series.index,
        )

    def process(self, context: dict) -> dict:
        """Add education level features to the DataFrame.

        :param context: current pipeline context shared between all handlers
        :type context: dict
        :return: updated context with education features
        :rtype: dict
        """
        print("\nCOMPLEX FEATURES...")

        df: pd.DataFrame = context["df"]
        df = df.copy()

        edu_features = self._parse_education_level(df["Образование и ВУЗ"])
        df = pd.concat([df, edu_features], axis=1)

        if self.keep_raw:
            df["raw_education"] = df["Образование и ВУЗ"]

        context["df"] = df
        print("Done")
        return context
