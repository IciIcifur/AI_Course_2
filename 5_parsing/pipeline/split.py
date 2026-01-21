from .base import Handler


class SplitTargetHandler(Handler):
    def __init__(self, target_column: str, next_handler=None):
        super().__init__(next_handler)
        self.target_column = target_column

    def process(self, context: dict) -> dict:
        print('\nSPLITTING X AND Y...')
        df = context["df"]

        context["y"] = df[self.target_column].values
        context["X"] = df.drop(columns=[self.target_column]).values

        print('Done')
        return context
