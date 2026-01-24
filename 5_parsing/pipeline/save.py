from pathlib import Path

import numpy as np

from .base import Handler


class NumpySaver(Handler):
    """Save feature matrix X and target vector y as NumPy .npy files."""

    def __init__(self, output_dir: Path, next_handler=None):
        """Initialize saver with output directory.

        :param output_dir: directory where .npy files will be written
        :type output_dir: Path
        :param next_handler: next handler to be executed in the pipeline
        :type next_handler: Handler or None
        """
        super().__init__(next_handler)
        self.output_dir = output_dir

    def process(self, context: dict) -> dict:
        """Save X and y from context to x_data.npy and y_data.npy.

        :param context: current pipeline context shared between all handlers
        :type context: dict
        :return: unchanged context after saving
        :rtype: dict
        """
        print("\nSAVING RESULTS...")

        x_path = self.output_dir / "x_data.npy"
        y_path = self.output_dir / "y_data.npy"

        np.save(x_path, context["X"])
        np.save(y_path, context["y"])

        print(f"Saved X to {x_path}")
        print(f"Saved y to {y_path}")
        print("Done")
        return context
