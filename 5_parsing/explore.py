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
    'Липецк , не готов к переезду , готов к редким командировкам'
    -> ('Липецк', reloc, business_trips_level)

    business_trips_level ∈ {'none', 'rare', 'regular', 'unknown'}.
    """
    s = series.astype(str)

    cities: list[str] = []
    reloc_flags: list[int] = []
    trips_levels: list[str] = []

    for value in s:
        raw = str(value)
        lower = raw.lower()

        # Разбиваем по запятым на логические куски:
        # [город, (метро), статус переезда, статус командировок]
        parts_raw = [p.strip() for p in raw.split(",") if p.strip()]
        parts_lower = [p.lower() for p in parts_raw]

        # --- Город ---
        city = parts_raw[0] if parts_raw else ""
        cities.append(city)

        # --- Переезд ---
        reloc = False
        for part in parts_lower:
            if "переезд" in part:
                # если явно написано 'не готов(а) к переезду' — считаем не готов
                if "не готов" in part or "не готова" in part:
                    reloc = False
                # есл�� 'готов(а) к переезду' и нет 'не' — считаем готов
                elif "готов к переезду" in part or "готова к переезду" in part:
                    reloc = True
        reloc_flags.append(int(reloc))

        # --- Командировки ---
        level = "unknown"
        found_trips = False

        for part in parts_lower:
            if "командиров" not in part:
                continue

            found_trips = True

            if "не готов" in part or "не готова" in part:
                level = "none"
                break  # это самый жёсткий вариант, можно сразу выходить
            if "редк" in part:
                level = "rare"
                # не выходим, вдруг в другом токене есть 'не готов', но это уже редкий случай
                continue
            if "готов" in part or "готова" in part:
                # если пока ничего лучше не нашли, считаем regular
                if level == "unknown":
                    level = "regular"

        if not found_trips:
            level = "unknown"

        trips_levels.append(level)

    city_series = pd.Series(cities, index=series.index)
    reloc_series = pd.Series(reloc_flags, index=series.index, dtype="int64")
    trips_series = pd.Series(trips_levels, index=series.index, dtype="string")

    return city_series, reloc_series, trips_series


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
        "school",
    ]
    level = np.select(conditions, choices, default="unknown")
    return pd.Series(level, index=series.index, dtype="string")


def parse_education_last_year(series: pd.Series) -> pd.Series:
    """
    Из 'Образование и ВУЗ' вытаскиваем последний (максимальный) год (4 цифры).
    Это условно год последнего (самого свежего) образования.
    """
    s = series.astype(str)
    years = s.str.findall(r"\b(19\d{2}|20\d{2})\b")

    def max_year(lst):
        if not lst:
            return np.nan
        return max(int(y) for y in lst)

    return years.apply(max_year).astype("float")


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


# ---------- Нормализация должности ----------

def normalize_position(series: pd.Series) -> pd.Series:
    """
    Должность в lower-case, с убранными лишними пробелами.
    Никаких ролей, просто нормализованная текстовая строка.
    """
    s = series.fillna("").astype(str).str.strip().str.lower()

    s = s.str.replace(r"\s*,\s*", ", ", regex=True)
    s = s.str.replace(r"\s*/\s*", " / ", regex=True)
    return s


# ---------- Главная функция аналитики ----------

def main() -> None:
    csv_path = Path("../data/input/hh.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    pd.set_option("display.max_colwidth", 324)
    pd.set_option("display.max_columns", None)

    print("\n=== PARSED FEATURES PREVIEW ===")

    sex, age = parse_sex_age(df["Пол, возраст"])
    salary = parse_salary(df["ЗП"])
    city, reloc, trips = parse_city_mobility(df["Город"])
    exp_years = parse_experience_years(df["Опыт (двойное нажатие для полной версии)"])
    edu_level = parse_education_level(df["Образование и ВУЗ"])
    edu_last_year = parse_education_last_year(df["Образование и ВУЗ"])
    has_car = parse_has_car(df["Авто"])
    schedule_norm = parse_schedule(df["График"])
    position_norm = normalize_position(df["Ищет работу на должность:"])
    last_position_norm = normalize_position(df["Последеняя/нынешняя должность"])

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
            "education_last_year": edu_last_year,
            "has_car": has_car,
            "schedule": schedule_norm,
            "position": position_norm,
            "last_position": last_position_norm,
            "raw_last_position": df["Последеняя/нынешняя должность"],
            "last_work": df["Последенее/нынешнее место работы"],
            "raw_education": df["Образование и ВУЗ"],
        }
    )

    print(parsed_df.tail(10))
    print("\nDtypes of parsed features:")
    print(parsed_df.dtypes)


if __name__ == "__main__":
    main()
