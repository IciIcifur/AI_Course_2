from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- Функции парсинга признаков ----------

def parse_sex_age(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    'Мужчина ,  42 года , родился 6 октября 1976' -> ('Мужчина', 42)
    """
    s = series.astype(str)

    sex = s.str.extract(r"^(Мужчина|Женщина)", expand=False)

    s_lower = s.str.lower()
    age_str = s_lower.str.extract(r"(\d+)\s*(?:год|года|лет)", expand=False)
    age = pd.to_numeric(age_str, errors="coerce")

    return sex, age


def parse_salary(series: pd.Series) -> pd.Series:
    """
    '27 000 руб.' -> (27000, todo: currency)
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
    -> ('Липецк', 0/1, business_trips_level)

    business_trips_level ∈ {'none', 'rare', 'regular', 'unknown'}.
    """
    s = series.astype(str)
    s_lower = s.str.lower()

    city = s.str.split(",", n=1).str[0].str.strip()

    reloc_ready = s_lower.str.contains("готов к переезду", na=False) | s_lower.str.contains("готова к переезду",
                                                                                            na=False)
    reloc_not_ready = s_lower.str.contains("не готов к переезду", na=False) | s_lower.str.contains(
        "не готова к переезду", na=False)
    reloc = reloc_ready & ~reloc_not_ready

    has_trips = s_lower.str.contains("командиров", na=False)
    has_not = s_lower.str.contains("не", na=False)
    has_rare = s_lower.str.contains("редк", na=False) # todo: doesn t work

    cond_none = has_not
    cond_rare = has_rare & ~has_not
    cond_regular = has_trips & ~cond_none & ~cond_rare

    trips_level = np.select(
        [cond_none, cond_rare, cond_regular],
        ["none", "rare", "regular"],
        default="unknown",
    )
    trips_level = pd.Series(trips_level, index=series.index, dtype="string")

    return city, reloc.astype(int), trips_level


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
    total_years = total_years.replace(0, np.nan)
    return total_years


def parse_education_level(series: pd.Series) -> pd.Series:
    """
    Из 'Образование и ВУЗ' выбираем уровень образования.
    Нормализуем в небольшой словарь категорий.
    todo: education_level, education_degree
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


# ---------- Нормализация графика ----------

def normalize_schedule_token(token: str) -> str:
    """
    Приводим различные варианты графика к фиксированным категориям:
    fullday, flexible, remote, shifts, rotation, other.
    """
    t = token.strip().lower()
    if t in ("полный день", "full day"):
        return "fullday"
    if t in ("гибкий график", "flexible schedule"):
        return "flexible"
    if t in ("удаленная работа", "remote working"):
        return "remote"
    if t in ("сменный график", "shift schedule"):
        return "shifts"
    if t in ("вахтовый метод", "rotation based work"):
        return "rotation"
    return "other"


def parse_schedule(series: pd.Series) -> pd.Series:
    """
    'удаленная работа, полный день, вахтовый метод' ->
    категориальный признак, в котором мы храним
    МНОЖЕСТВО нормализованных токенов как строку, например:
      'fullday|remote|rotation'
    Чтобы по-прежнему иметь один столбец; в пайплайне
    можно будет развернуть это в multi-hot.
    """
    s = series.fillna("").astype(str)
    normed_values = []

    for value in s:
        parts = [p for p in value.split(",") if p.strip()]
        normed = [normalize_schedule_token(p) for p in parts]
        normed_set = sorted(set(normed))
        normed_values.append("|".join(normed_set) if normed_set else "")

    return pd.Series(normed_values, index=series.index, dtype="string")


# ---------- Анализ многозначных (через запятую) колонок ----------

def inspect_multivalue_column(series: pd.Series, name: str | None = None) -> None:
    """
    Разбивает значения по запятым, триммит пробелы, считает частоты токенов.
    Печатает все токены.
    """
    s = series.dropna().astype(str)
    tokens: Counter[str] = Counter()

    for value in s:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        tokens.update(parts)

    print("\n=== MULTIVALUE ANALYSIS:", name or series.name, "===")
    for token, cnt in tokens.most_common():
        print(f"{token!r}: {cnt}")


# ---------- Анализ должностей ----------

def inspect_positions(series: pd.Series) -> None:
    """
    Выводит ВСЕ уникальные должности в lower/strip с их частотами.
    """
    s = series.fillna("").astype(str).str.lower().str.strip()
    counter = Counter(s)
    print("\n=== ALL POSITIONS (lowercased) ===")
    for value, cnt in counter.most_common():
        print(f"{value!r}: {cnt}")


# ---------- Главная функция аналитики ----------

def main() -> None:
    csv_path = Path("../data/input/hh.csv")  # пока жёстко, потом заменим на аргумент
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)


    pd.set_option("display.max_colwidth", 324)
    pd.set_option("display.max_columns", None)

    # ---------- Распарсим ключевые признаки ----------

    print("\n=== PARSED FEATURES PREVIEW ===")

    sex, age = parse_sex_age(df["Пол, возраст"])
    salary = parse_salary(df["ЗП"])
    city, reloc, trips_level = parse_city_mobility(df["Город"])
    exp_years = parse_experience_years(df["Опыт (двойное нажатие для полной версии)"])
    edu_level = parse_education_level(df["Образование и ВУЗ"])
    has_car = parse_has_car(df["Авто"])
    schedule_norm = parse_schedule(df["График"])

    parsed_df = pd.DataFrame(
        {
            "sex": sex,
            "age": age,
            "salary": salary,
            "city": city,
            "relocation": reloc,
            "business_trips_level": trips_level,
            "experience_years": exp_years,
            "education_level": edu_level,
            "has_car": has_car,
            "schedule_normalized": schedule_norm,
            "raw_position": df["Ищет работу на должность:"],
            "raw_last_position": df["Последеняя/нынешняя должность"],
            "last_work": df["Последенее/нынешнее место работы"],
        }
    )

    print(parsed_df.head(10))

    # ---------- Анализ multi-value колонок и всех должностей ----------
    inspect_positions(df["Ищет работу на должность:"])


if __name__ == "__main__":
    main()
