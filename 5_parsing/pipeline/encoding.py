import pandas as pd
from .base import Handler


class EncodingHandler(Handler):
    def process(self, context: dict) -> dict:
        print('\nENCODING COLUMNS...')

        df = context["df"]
        cat_cols = df.select_dtypes(include=["object"]).columns

        for col in cat_cols:
            if df[col].nunique() > 50:
                df = df.drop(columns=[col])

        cat_cols = df.select_dtypes(include=["object"]).columns

        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        context["df"] = df
        print('Done')
        return context
