from .base import Handler


class DataCleaningHandler(Handler):
    def process(self, context: dict) -> dict:
        print('\nCLEANSING DATA...')

        df = context["df"]

        df = df.replace("\ufeff", "", regex=True)
        df = df.replace("\xa0", " ", regex=True)

        df = df.drop_duplicates()

        context["df"] = df
        print('Done')
        return context
