from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import Handler


class CSVLoader(Handler):
    """Load raw CSV data into the pipeline context."""

    def __init__(self, path: str, next_handler=None):
        """Initialize loader with path to the input CSV file.

        :param path: path to the input CSV file
        :type path: str
        :param next_handler: next handler to be executed in the pipeline
        :type next_handler: Handler or None
        """
        super().__init__(next_handler)
        self.path = path

    def process(self, context: dict) -> dict:
        """Read CSV into a DataFrame and store it in the context.

        :param context: current pipeline context shared between all handlers
        :type context: dict
        :return: updated context with loaded DataFrame
        :rtype: dict
        """
        print("\nLOADING DATA...")

        csv_path = Path(self.path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        for col in ("Unnamed: 0", "Unnamed: 0.1"):
            if col in df.columns:
                df = df.drop(columns=[col])

        context["df"] = df
        print("Done")
        return context
