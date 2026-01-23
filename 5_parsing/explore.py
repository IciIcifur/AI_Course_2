from pathlib import Path
import re
from typing import Tuple

import numpy as np
import pandas as pd


# ---------- Функции парсинга признаков ----------

def parse_sex_age(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    'Мужчина ,  42 года , родился 6 октября 1976' -> ('Мужчина', 42)
    """
    s = series.astype(str)

    # пол
    sex = s.str.extract(r"^(Мужчина|Женщина)", expand=False)

    s_lower = s.str.lower()

    age_str = s_lower.str.extract(r"(\d+)\s*(?:год|года|лет)", expand=False)
    age = pd.to_numeric(age_str, errors="coerce")

    return sex, age


def parse_salary(series: pd.Series) -> pd.Series:
    """
    '27 000 руб.' -> 27000
    Убираем пробелы и неразрывные пробелы, оставляем только цифры.
    """
    cleaned = (
        series.astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.replace(" ", "", regex=False)
    )
    numeric_str = cleaned.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(numeric_str, errors="coerce")


def parse_city_mobility(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    'Липецк , не готов к переезду , не готов к командировкам'
    -> ('Липецк', 0, 0)
    """
    s = series.astype(str)

    city = s.str.split(",", n=1).str[0].str.strip()

    reloc = s.str.contains("готов к переезду", case=False, na=False) & \
            ~s.str.contains("не готов к переезду", case=False, na=False)

    trips = s.str.contains("готов к командировкам", case=False, na=False) & \
            ~s.str.contains("не готов к командировкам", case=False, na=False)

    return city, reloc.astype(int), trips.astype(int)


def parse_experience_years(series: pd.Series) -> pd.Series:
    """
    Берем первую строку поля опыта и пытаемся оценить годы опыта.
    Пример: '13 лет 2 месяца' -> 13 + 2/12.
    """
    s = series.astype(str).str.split("\n").str[0]

    years = s.str.extract(r"(\d+)\s*(?:год|года|лет)", expand=False)
    months = s.str.extract(r"(\d+)\s*(?:месяц|месяца|месяцев)", expand=False)

    years_num = pd.to_numeric(years, errors="coerce").fillna(0)
    months_num = pd.to_numeric(months, errors="coerce").fillna(0)

    total_years = years_num + months_num / 12.0
    # если не нашли ни лет, ни месяцев — ставим NaN
    total_years = total_years.replace(0, np.nan)
    return total_years


def parse_education_level(series: pd.Series) -> pd.Series:
    """
    Из 'Образование и ВУЗ' выдираем уровень образования.
    Н��рмализуем в небольшой словарь категорий.
    """
    s = series.astype(str).str.lower()

    conditions = [
        s.str.contains("магистр", na=False),
        s.str.contains("бакалавр", na=False),
        s.str.contains("высшее", na=False),
        s.str.contains("среднее профессиональное", na=False)
        | s.str.contains("среднее специальное", na=False),
        s.str.contains("среднее общее", na=False),
    ]
    choices = [
        "master",
        "bachelor",
        "higher",
        "vocational",
        "secondary",
    ]
    level = np.select(conditions, choices, default="unknown")
    return pd.Series(level, index=series.index, dtype="string")


def parse_has_car(series: pd.Series) -> pd.Series:
    """
    'Имеется собственный автомобиль' -> 1, остальное -> 0.
    """
    s = series.astype(str).str.lower()
    has_car = s.str.contains("имеется собственный автомобиль", na=False)
    return has_car.astype(int)


# ---------- Главная функция аналитики ----------

def main() -> None:
    csv_path = Path("../data/input/hh.csv")  # пока жёстко, потом заменим на аргумент
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print("=== head(10) ===")
    pd.set_option("display.max_colwidth", 324)
    pd.set_option("display.max_columns", None)

    print(df.iloc[:, 1:].head(10))

    '''
    print("\n=== columns ===")
    print(df.columns.tolist())

    print("\n=== dtypes ===")
    print(df.dtypes)
    

    # Немного подробностей по текстовым колонкам
    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    print("\n=== text columns sample ===")
    print(text_cols)

    sample_cols = text_cols[:5]
    for col in sample_cols:
        print(f"\n--- Column: {col} ---")
        print("Sample value:")
        print(df[col].iloc[0])
        print(f"nunique: {df[col].nunique(dropna=True)}")

    '''
    # ---------- Распарсим ключевые признаки ----------

    print("\n=== PARSED FEATURES PREVIEW ===")

    sex, age = parse_sex_age(df["Пол, возраст"])
    salary = parse_salary(df["ЗП"])
    city, reloc, trips = parse_city_mobility(df["Город"])
    exp_years = parse_experience_years(df["Опыт (двойное нажатие для полной версии)"])
    edu_level = parse_education_level(df["Образование и ВУЗ"])
    has_car = parse_has_car(df["Авто"])

    pd.set_option("display.max_colwidth", None)

    parsed_df = pd.DataFrame(
        {
            "sex": sex,
            "age": age,
            "salary": salary,
            "city": city,
            "relocation": reloc,
            "business_trips": trips,
            "experience_years": exp_years,
            "education_level": edu_level,
            "has_car": has_car,
            "raw_position": df["Ищет работу на должность:"],
            "raw_last_position": df["Последеняя/нынешняя должность"],
            "raw_last_work": df["Последенее/нынешнее место работы"],
        }
    )

    print(parsed_df.head(10))
    print("\nDtypes of parsed features:")
    print(parsed_df.dtypes)


if __name__ == "__main__":
    main()
