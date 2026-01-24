import pandas as pd

from .base import Handler


class DataCleaningHandler(Handler):
    def process(self, context: dict) -> dict:
        print("\nCLEANSING DATA...")

        df: pd.DataFrame = context["df"]

        # Убираем BOM и неразрывные пробелы
        df = df.replace("\ufeff", "", regex=True)
        df = df.replace("\xa0", " ", regex=True)

        # Дроп дубликатов
        df = df.drop_duplicates()

        context["df"] = df
        print("Done")
        return context
