import pandas as pd

from .base import Handler


class DataCleaningHandler(Handler):
    """Perform basic text cleanup and deduplication of the raw DataFrame."""

    def process(self, context: dict) -> dict:
        """Clean raw text artifacts and drop duplicate rows.

        :param context: current pipeline context shared between all handlers
        :type context: dict
        :return: updated context with cleaned DataFrame
        :rtype: dict
        """
        print("\nCLEANSING DATA...")

        df: pd.DataFrame = context["df"]

        df = df.replace("\ufeff", "", regex=True)
        df = df.replace("\xa0", " ", regex=True)

        df = df.drop_duplicates()

        context["df"] = df
        print("Done")
        return context
