from __future__ import annotations

import numpy as np
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

        df: pd.DataFrame = context["df"].copy()

        if self.target_column not in df.columns:
            raise KeyError(f"Target column not found: {self.target_column}")

        y = df[self.target_column].to_numpy()
        x_df = df.drop(columns=[self.target_column])

        for col in x_df.columns:
            if not pd.api.types.is_numeric_dtype(x_df[col]):
                x_df[col] = pd.to_numeric(x_df[col], errors="coerce")

        x = x_df.to_numpy(dtype=np.float64)

        context["X"] = x
        context["y"] = y

        print("Done")
        return context
