import pandas as pd
from .base import Handler


class CSVLoader(Handler):
    def __init__(self, path: str, next_handler=None):
        super().__init__(next_handler)
        self.path = path

    def process(self, context: dict) -> dict:
        print('\nLOADING DATA...')
        df = pd.read_csv(self.path, index_col=0)
        context["df"] = df
        print('Done')
        return context
