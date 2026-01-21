from .base import Handler


class DataCleaningHandler(Handler):
    def process(self, context: dict) -> dict:
        df = context["df"]

        df = df.replace("\ufeff", "", regex=True)
        df = df.replace("\xa0", " ", regex=True)

        df = df.drop_duplicates()

        context["df"] = df
        return context
