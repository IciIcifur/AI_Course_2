import pandas as pd

from .base import Handler


class CategoryHandler(Handler):
    """Extract categorical profile and location features from raw HH.ru data."""

    def _parse_city_mobility(
            self, series: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Parse city, relocation flag and business trips level from 'Город' column.

        :param series: source column with raw city and mobility information
        :type series: pd.Series
        :return: tuple of (city, relocation_flag, business_trips) series
        :rtype: tuple[pd.Series, pd.Series, pd.Series]
        """
        s = series.astype(str)

        cities: list[str] = []
        reloc_flags: list[int] = []
        trips_levels: list[str] = []

        for value in s:
            raw = str(value)
            parts_raw = [p.strip() for p in raw.split(",") if p.strip()]
            parts_lower = [p.lower() for p in parts_raw]

            city = parts_raw[0] if parts_raw else ""
            cities.append(city)

            reloc = False
            for part in parts_lower:
                if "переезд" in part:
                    if "не готов" in part or "не готова" in part:
                        reloc = False
                    elif "готов к переезду" in part or "готова к переезду" in part:
                        reloc = True
            reloc_flags.append(int(reloc))

            level = "unknown"
            found_trips = False

            for part in parts_lower:
                if "командиров" not in part:
                    continue

                found_trips = True

                if "не готов" in part or "не готова" in part:
                    level = "none"
                    break
                if "редк" in part:
                    level = "rare"
                    continue
                if "готов" in part or "готова" in part:
                    if level == "unknown":
                        level = "regular"

            if not found_trips:
                level = "unknown"

            trips_levels.append(level)

        city_series = pd.Series(cities, index=series.index)
        reloc_series = pd.Series(reloc_flags, index=series.index, dtype="int64")
        trips_series = pd.Series(trips_levels, index=series.index, dtype="string")

        return city_series, reloc_series, trips_series

    def _normalize_schedule_token(self, token: str) -> str:
        """Normalize raw schedule token to a compact category code.

        :param token: raw schedule token text
        :type token: str
        :return: normalized schedule token
        :rtype: str
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

    def _parse_schedule(self, series: pd.Series) -> pd.Series:
        """Parse and normalize work schedule from 'График' column.

        :param series: source column with raw schedule description
        :type series: pd.Series
        :return: series with '|' separated normalized schedule tokens
        :rtype: pd.Series
        """
        s = series.fillna("").astype(str)
        normed_values: list[str] = []

        for value in s:
            parts = [p for p in value.split(",") if p.strip()]
            normed = [self._normalize_schedule_token(p) for p in parts]
            normed_set = sorted(set(normed))
            normed_values.append("|".join(normed_set) if normed_set else "")

        return pd.Series(normed_values, index=series.index, dtype="string")

    def _parse_sex(self, series: pd.Series) -> pd.Series:
        """Parse sex from the 'Пол, возраст' column.

        :param series: source column with raw sex and age information
        :type series: pd.Series
        :return: series with parsed sex values
        :rtype: pd.Series
        """
        s = series.astype(str)
        sex = s.str.extract(r"^(Мужчина|Женщина)", expand=False)
        return sex

    def process(self, context: dict) -> dict:
        """Add categorical profile and location features to the DataFrame.

        :param context: current pipeline context shared between all handlers
        :type context: dict
        :return: updated context with added categorical features
        :rtype: dict
        """
        print("\nCATEGORY FEATURES...")

        df: pd.DataFrame = context["df"]

        sex = self._parse_sex(df["Пол, возраст"])
        city, reloc, trips = self._parse_city_mobility(df["Город"])
        schedule_norm = self._parse_schedule(df["График"])

        df = df.copy()
        df["sex"] = sex
        df["city"] = city
        df["relocation"] = reloc
        df["business_trips"] = trips
        df["schedule"] = schedule_norm

        context["df"] = df
        print("Done")
        return context
