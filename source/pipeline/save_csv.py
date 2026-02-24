from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import Handler


class DataFrameSaver(Handler):
    """Save current DataFrame from context to a CSV file."""

    def __init__(self, output_path: Path, next_handler=None, index: bool = False):
        """Create saver.

        :param output_path: where to save CSV
        :type output_path: Path
        :param next_handler: next handler in pipeline
        :type next_handler: Handler or None
        :param index: whether to store dataframe index in CSV
        :type index: bool
        """
        super().__init__(next_handler)
        self.output_path = output_path
        self.index = index

    def process(self, context: dict) -> dict:
        """Save context['df'] to CSV."""
        print(f"\nSAVING DF TO CSV: {self.output_path} ...")

        df: pd.DataFrame = context["df"]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=self.index)

        print("Done")
        return context
