"""French market reference data: regions, local time (with DST), weekend, holidays, sales periods."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

TZ = "Europe/Paris"

# The 13 metropolitan regions (2016 reform) and their departments: 96 codes incl. Corsica 2A/2B.
REGIONS = {
    "Auvergne-Rhône-Alpes": ["01", "03", "07", "15", "26", "38", "42", "43", "63", "69", "73", "74"],
    "Bourgogne-Franche-Comté": ["21", "25", "39", "58", "70", "71", "89", "90"],
    "Bretagne": ["22", "29", "35", "56"],
    "Centre-Val de Loire": ["18", "28", "36", "37", "41", "45"],
    "Corse": ["2A", "2B"],
    "Grand Est": ["08", "10", "51", "52", "54", "55", "57", "67", "68", "88"],
    "Hauts-de-France": ["02", "59", "60", "62", "80"],
    "Île-de-France": ["75", "77", "78", "91", "92", "93", "94", "95"],
    "Normandie": ["14", "27", "50", "61", "76"],
    "Nouvelle-Aquitaine": ["16", "17", "19", "23", "24", "33", "40", "47", "64", "79", "86", "87"],
    "Occitanie": ["09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66", "81", "82"],
    "Pays de la Loire": ["44", "49", "53", "72", "85"],
    "Provence-Alpes-Côte d'Azur": ["04", "05", "06", "13", "83", "84"],
}
REGION_BY_DEPT = {d: r for r, depts in REGIONS.items() for d in depts}


def department(postcode: str | None) -> str | None:
    """Metropolitan department from a postcode; None for overseas (97x/98x) or invalid input."""
    if not isinstance(postcode, str) or not postcode.strip().isdigit() or len(postcode.strip()) != 5:
        return None
    cp = postcode.strip()
    if cp.startswith("20"):  # Corsica: 200xx-201xx = Corse-du-Sud, 202xx-206xx = Haute-Corse
        return "2A" if cp[:3] in ("200", "201") else "2B"
    return cp[:2] if cp[:2] in REGION_BY_DEPT else None


def region(dept: str | None) -> str:
    return REGION_BY_DEPT.get(dept, "Inconnue")


def easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def public_holidays(year: int) -> set[date]:
    e = easter(year)
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    return {date(year, m, d) for m, d in fixed} | {
        e + timedelta(days=1),
        e + timedelta(days=39),
        e + timedelta(days=50),
    }


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def sales_periods(year: int) -> list[tuple[date, date]]:
    """National soldes: 2nd Wednesday of January and last Wednesday of June, 4 weeks each
    (some border departments have other dates)."""
    winter = _nth_weekday(year, 1, 2, 2)
    summer = _last_weekday(year, 6, 2)
    return [(winter, winter + timedelta(days=27)), (summer, summer + timedelta(days=27))]


def black_friday(year: int) -> date:
    return _nth_weekday(year, 11, 3, 4) + timedelta(days=1)


def to_local(ts: pd.Series) -> pd.Series:
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert(TZ)


def calendar_features(local: pd.Series) -> pd.DataFrame:
    days = local.dt.tz_localize(None).dt.date
    years = sorted({d.year for d in days})
    holidays = set().union(*(public_holidays(y) for y in years))
    sales = [p for y in years for p in sales_periods(y)]
    bf = {y: black_friday(y) for y in years}
    return pd.DataFrame(
        {
            "is_weekend_fr": local.dt.dayofweek.isin([5, 6]).astype(int).to_numpy(),
            "is_holiday": np.array([d in holidays for d in days], dtype=int),
            "in_soldes": np.array([any(s <= d <= e for s, e in sales) for d in days], dtype=int),
            # Black Friday week-end through Cyber Monday.
            "black_friday": np.array([0 <= (d - bf[d.year]).days <= 3 for d in days], dtype=int),
            "pre_christmas": np.array([d.month == 12 and d.day <= 24 for d in days], dtype=int),
        },
        index=local.index,
    )
