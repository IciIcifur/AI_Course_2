from .base import Handler


class FeatureSelectionHandler(Handler):
    def process(self, context: dict) -> dict:
        print('\nEXTRACTING MEANINGFUL COLUMNS...')
        df = context["df"]

        print(df.head())
        print(df.columns)

        keep_columns = [
            "Пол, возраст",
            "ЗП",
            "График работы",
            "Занятость",
            "Образование",
            "Авто",
            "Желаемая зарплата",
        ]

        df = df[keep_columns]

        context["df"] = df
        return context
