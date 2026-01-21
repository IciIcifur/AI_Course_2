import numpy as np
from .base import Handler


class NumpySaver(Handler):
    def process(self, context: dict) -> dict:
        print('\nSAVING RESULTS...')

        np.save("data/output/x_data.npy", context["X"])
        np.save("data/output/y_data.npy", context["y"])

        print('Done')
        return context
