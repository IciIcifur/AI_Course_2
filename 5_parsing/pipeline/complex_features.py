import pandas as pd
import numpy as np
from .base import Handler


class ComplexHandler(Handler):
    """
    Обрабатывает колонку 'Образование и ВУЗ':
    - education_level
    - (raw_education остаётся для будущего анализа)
    """

    def __init__(self, next_handler=None, keep_raw: bool = True):
        super().__init__(next_handler)
        self.keep_raw = keep_raw

    def _parse_education_level(self, series: pd.Series) -> pd.Series:
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
        print("\nEDUCATION FEATURES...")

        df: pd.DataFrame = context["df"]

        edu_level = self._parse_education_level(df["Образование и ВУЗ"])
        df = df.copy()
        df["education_level"] = edu_level

        if self.keep_raw:
            df["raw_education"] = df["Образование и ВУЗ"]

        context["df"] = df
        print("Done")
        return context
