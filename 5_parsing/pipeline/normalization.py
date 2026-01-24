import pandas as pd

from .base import Handler


class NormalizeHandler(Handler):
    """
    Обработка должностей:
    - нормализуем текущую и последнюю должность
    - при этом сырые текстовые колонки по должности дропаем
      (образование остаётся в EducationHandler)
    """

    def _normalize_position(self, series: pd.Series) -> pd.Series:
        s = series.fillna("").astype(str).str.strip().str.lower()
        s = s.str.replace(r"\s*,\s*", ", ", regex=True)
        s = s.str.replace(r"\s*/\s*", " / ", regex=True)
        return s

    def process(self, context: dict) -> dict:
        print("\nPOSITION FEATURES...")

        df: pd.DataFrame = context["df"]

        position_norm = self._normalize_position(df["Ищет работу на должность:"])
        last_position_norm = self._normalize_position(df["Последеняя/нынешняя должность"])

        df = df.copy()
        df["position"] = position_norm
        df["last_position"] = last_position_norm

        # Можно оставить last_work как raw-текст, если считаешь полезным
        df["last_work"] = df["Последенее/нынешнее место работы"]

        # Дропаем сырые должностные поля (оставляем только нормализованные)
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
