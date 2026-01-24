from pathlib import Path

import numpy as np

from .base import Handler


class NumpySaver(Handler):
    def __init__(self, output_dir: Path, next_handler=None):
        super().__init__(next_handler)
        self.output_dir = output_dir

    def process(self, context: dict) -> dict:
        print("\nSAVING RESULTS...")

        x_path = self.output_dir / "x_data.npy"
        y_path = self.output_dir / "y_data.npy"

        np.save(x_path, context["X"])
        np.save(y_path, context["y"])

        print(f"Saved X to {x_path}")
        print(f"Saved y to {y_path}")
        print("Done")
        return context
