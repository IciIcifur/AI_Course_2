import pandas as pd

from .base import Handler


class SplitTargetHandler(Handler):
    def __init__(self, target_column: str, next_handler=None):
        super().__init__(next_handler)
        self.target_column = target_column

    def process(self, context: dict) -> dict:
        print("\nSPLITTING X AND Y...")

        df: pd.DataFrame = context["df"]

        pd.set_option("display.max_colwidth", 324)
        pd.set_option("display.max_columns", None)
        print(df.head())

        y = df[self.target_column].values
        X = df.drop(columns=[self.target_column]).values

        context["y"] = y
        context["X"] = X

        print("Done")
        return context
