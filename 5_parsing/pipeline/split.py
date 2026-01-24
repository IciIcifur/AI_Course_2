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
        print("\nDF HEAD BEFORE SPLIT:")
        print(df.head())

        print("\nDTYPES BEFORE SPLIT:")
        print(df.dtypes)

        print("\nNaN COUNTS PER COLUMN:")
        print(df.isna().sum())

        obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        print("\nObject/string columns before split:", obj_cols)

        for col in df.columns:
            if col == self.target_column:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        y = df[self.target_column].to_numpy()
        X = df.drop(columns=[self.target_column]).to_numpy(dtype="float64")

        context["y"] = y
        context["X"] = X

        print("Done")
        return context
