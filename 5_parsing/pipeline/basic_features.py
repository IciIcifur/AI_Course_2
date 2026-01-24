import numpy as np
import pandas as pd

from .base import Handler


class BasicHandler(Handler):
    """
    Базовые числовые и бинарные признаки:
    salary, currency, age, experience_years, education_last_year, has_car.
    """

    # ====== Вспомогательные методы-парсеры ======

    def _parse_sex_age(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        s = series.astype(str)
        sex = s.str.extract(r"^(Мужчина|Женщина)", expand=False)
        s_lower = s.str.lower()
        age_str = s_lower.str.extract(r"(\d+)\s*(?:год|года|лет)", expand=False)
        age = pd.to_numeric(age_str, errors="coerce")
        return sex, age.astype("Int64")

    def _parse_salary_and_currency(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        s = series.astype(str).str.replace("\xa0", " ", regex=False)

        numeric_str = (
            s.str.replace(" ", "", regex=False)
            .str.extract(r"(\d+)", expand=False)
        )
        salary = pd.to_numeric(numeric_str, errors="coerce")

        tail = (
            s.str.replace(" ", "", regex=False)
            .str.extract(r"\d+(.*)", expand=False)
            .fillna("")
            .str.strip()
            .str.lower()
        )

        def extract_currency(text: str) -> str:
            if not text:
                return "unknown"

            import re
            tokens = [t for t in re.split(r"[^0-9a-zA-Zа-яА-Я]+", text) if t]
            if not tokens:
                return "unknown"
            return tokens[-1]

        currency = tail.apply(extract_currency).astype("string")

        return salary, currency

    def _parse_experience_years(self, series: pd.Series) -> pd.Series:
        s = series.astype(str).str.split("\n").str[0]

        years = s.str.extract(r"(\d+)\s*(?:год|года|лет)", expand=False)
        months = s.str.extract(r"(\d+)\s*(?:месяц|месяца|месяцев)", expand=False)

        years_num = pd.to_numeric(years, errors="coerce").fillna(0)
        months_num = pd.to_numeric(months, errors="coerce").fillna(0)

        total_years = years_num + months_num / 12.0
        total_years = total_years.replace(0, np.nan)
        return total_years.round(2)

    def _parse_education_last_year(self, series: pd.Series) -> pd.Series:
        s = series.astype(str)
        years = s.str.findall(r"\b(19\d{2}|20\d{2})\b")

        def max_year(lst):
            if not lst:
                return np.nan
            return max(int(y) for y in lst)

        return years.apply(max_year).astype("Int64")

    def _parse_has_car(self, series: pd.Series) -> pd.Series:
        s = series.astype(str).str.lower()
        has_car = s.str.contains("имеется собственный автомобиль", na=False)
        return has_car.astype(int)

    # ====== Основной метод ======

    def process(self, context: dict) -> dict:
        print("\nBASIC NUMERIC FEATURES...")

        df: pd.DataFrame = context["df"]

        sex, age = self._parse_sex_age(df["Пол, возраст"])
        salary, currency = self._parse_salary_and_currency(df["ЗП"])
        exp_years = self._parse_experience_years(df["Опыт (двойное нажатие для полной версии)"])
        edu_last_year = self._parse_education_last_year(df["Образование и ВУЗ"])
        has_car = self._parse_has_car(df["Авто"])

        df = df.copy()
        df["age"] = age
        df["salary"] = salary
        df["currency"] = currency
        df["experience_years"] = exp_years
        df["education_last_year"] = edu_last_year
        df["has_car"] = has_car
        df["sex"] = sex

        context["df"] = df
        print("Done")
        return context
