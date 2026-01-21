from .base import Handler
import pandas as pd


class FeatureSelectionHandler(Handler):
    """
    Универсальный сплит столбцов по запятым
    Обработка специальных колонок:
    - Опыт: берем только нулевой элемент (до первой запятой)
    - Образование и ВУЗ: пропускаем
    """

    def __init__(
            self,
            dirty_cols=None,
            structured_cols=None,  # колонки, которые можно безопасно делить
            next_handler=None
    ):
        super().__init__(next_handler)
        self.dirty_cols = dirty_cols if dirty_cols else ['Образование и ВУЗ', 'Опыт (двойное нажатие для полной версии)']
        self.structured_cols = structured_cols if structured_cols else ['Пол, возраст', 'Ищет работу на должность:',
                                                                        'Город', 'Занятость', 'График', 'Авто']

    def process(self, context: dict) -> dict:
        print('\nEXTRACTING MEANINGFUL COLUMNS...')
        df = context["df"]

        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_columns", None)

        print(df.columns)
        new_cols = {}

        for col in df.columns:
            if df[col].isnull().all():
                continue

            # Обработка опыта
            if col in self.dirty_cols:
                cleaned = df[col].astype(str).str.replace(r'[\n\r\t]+', ' ', regex=True).str.strip()
                # берем только текст до первой запятой
                new_cols[col] = cleaned #.str.split(',', n=1).str[0]
                continue

            # Для структурированных колонок делим по запятым и создаем n+1 колонок
            if col in self.structured_cols:
                split_df = df[col].astype(str).str.split(',', expand=True)
                split_df = split_df.apply(lambda x: x.str.strip())  # убрать пробелы
                split_df.columns = [f"{col}_{i}" for i in range(split_df.shape[1])]
                new_cols.update(split_df.to_dict(orient='series'))
                continue

            # Остальные колонки оставляем как есть
            new_cols[col] = df[col]

        df_new = pd.DataFrame(new_cols)

        print(df_new.head(1))
        print(df_new.columns)
        context['df'] = df_new
        return context
