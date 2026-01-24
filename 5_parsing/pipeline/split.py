import pandas as pd

from .base import Handler


class SplitTargetHandler(Handler):
    """Split the final DataFrame into feature matrix X and target vector y."""

    def __init__(self, target_column: str, next_handler=None):
        """Initialize splitter with the name of the target column.

        :param target_column: name of the target column to be used as y
        :type target_column: str
        :param next_handler: next handler to be executed in the pipeline
        :type next_handler: Handler or None
        """
        super().__init__(next_handler)
        self.target_column = target_column

    def process(self, context: dict) -> dict:
        """Split DataFrame into X and y and store them in the context.

        :param context: current pipeline context shared between all handlers
        :type context: dict
        :return: updated context with 'X' and 'y' arrays added
        :rtype: dict
        """
        print("\nSPLITTING X AND Y...")

        df: pd.DataFrame = context["df"]

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
