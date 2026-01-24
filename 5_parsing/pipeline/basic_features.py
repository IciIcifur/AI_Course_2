import numpy as np
import pandas as pd

from .base import Handler


class BasicHandler(Handler):
    """Extract basic numeric features from raw HH.ru data."""

    def _parse_sex_age(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Parse sex and age from the raw 'Пол, возраст' column.

        :param series: source column with raw sex and age information
        :type series: pd.Series
        :return: tuple of (sex, age) series
        :rtype: tuple[pd.Series, pd.Series]
        """
        s = series.astype(str)

        sex = s.str.extract(r"^(Мужчина|Женщина)", expand=False)

        mask_missing = sex.isna()
        if mask_missing.any():
            sex_en = s[mask_missing].str.extract(r"^(Male|Female)", expand=False)
            sex_en_mapped = sex_en.map({"Male": "Мужчина", "Female": "Женщина"})
            sex.loc[mask_missing] = sex_en_mapped

        s_lower = s.str.lower()

        age_str = s_lower.str.extract(r"(\d+)\s*(?:год|года|лет)", expand=False)

        mask_age_missing = age_str.isna()
        if mask_age_missing.any():
            age_str_en = s_lower[mask_age_missing].str.extract(r"(\d+)\s*(?:year|years)", expand=False)
            age_str.loc[mask_age_missing] = age_str_en

        age = pd.to_numeric(age_str, errors="coerce").astype("Int64")

        return sex, age

    def _parse_salary_and_currency(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Parse numeric salary amount and currency code from the 'ЗП' column.

        :param series: source column with raw salary text
        :type series: pd.Series
        :return: tuple of (salary, currency) series
        :rtype: tuple[pd.Series, pd.Series]
        """
        s = series.astype(str).str.replace("\xa0", " ", regex=False)

        numeric_str = (s.str.replace(" ", "", regex=False).str.extract(r"(\d+)", expand=False))
        salary = pd.to_numeric(numeric_str, errors="coerce")

        tail = (s.str.replace(" ", "", regex=False).str.extract(r"\d+(.*)", expand=False).fillna(
            "").str.strip().str.lower())

        def extract_currency(text: str) -> str:
            """Extract currency token from salary tail text."""
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
        """Parse total years of experience from raw experience column.

        :param series: source column with raw experience description
        :type series: pd.Series
        :return: total experience in years, rounded to two decimals
        :rtype: pd.Series
        """
        s = series.astype(str).str.split("\n").str[0]

        years = s.str.extract(r"(\d+)\s*(?:год|года|лет)", expand=False)
        months = s.str.extract(r"(\d+)\s*(?:месяц|месяца|месяцев)", expand=False)

        years_num = pd.to_numeric(years, errors="coerce").fillna(0)
        months_num = pd.to_numeric(months, errors="coerce").fillna(0)

        total_years = years_num + months_num / 12.0
        total_years = total_years.replace(0, np.nan)
        return total_years.round(2)

    def _parse_education_last_year(self, series: pd.Series) -> pd.Series:
        """Parse last education year from the 'Образование и ВУЗ' column.

        :param series: source column with raw education description
        :type series: pd.Series
        :return: series with the most recent education year
        :rtype: pd.Series
        """
        s = series.astype(str)
        years = s.str.findall(r"\b(19\d{2}|20\d{2})\b")

        def max_year(lst):
            if not lst:
                return np.nan
            return max(int(y) for y in lst)

        return years.apply(max_year).astype("Int64")

    def _parse_has_car(self, series: pd.Series) -> pd.Series:
        """Parse car ownership flag from the 'Авто' column.

        :param series: source column with raw car ownership text
        :type series: pd.Series
        :return: binary series indicating car ownership
        :rtype: pd.Series
        """
        s = series.astype(str).str.lower()
        has_car = s.str.contains("имеется собственный автомобиль", na=False)
        return has_car.astype(int)

    def process(self, context: dict) -> dict:
        """Add basic numeric features to the DataFrame.

        :param context: current pipeline context shared between all handlers
        :type context: dict
        :return: updated context with added basic features
        :rtype: dict
        """
        print("\nBASIC FEATURES...")

        df: pd.DataFrame = context["df"]

        sex, age = self._parse_sex_age(df["Пол, возраст"])
        salary, currency = self._parse_salary_and_currency(df["ЗП"])
        exp_years = self._parse_experience_years(
            df["Опыт (двойное нажатие для полной версии)"]
        )
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
