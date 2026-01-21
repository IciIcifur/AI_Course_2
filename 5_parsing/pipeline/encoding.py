import pandas as pd
from .base import Handler


class EncodingHandler(Handler):
    def process(self, context: dict) -> dict:
        df = context["df"]

        cat_cols = df.select_dtypes(include=["object"]).columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        context["df"] = df
        return context
