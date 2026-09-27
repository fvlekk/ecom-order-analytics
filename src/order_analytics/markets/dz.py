"""Algerian market reference data: wilayas, regions, local time, weekend, Ramadan/Eid calendar."""

import unicodedata

import numpy as np
import pandas as pd

TZ = "Africa/Algiers"

# Same order and spelling as bizz/client/src/data/wilayas.js: index + 1 = official wilaya code.
WILAYAS = [
    "Adrar", "Chlef", "Laghouat", "Oum El Bouaghi", "Batna", "Béjaïa", "Biskra",
    "Béchar", "Blida", "Bouira", "Tamanrasset", "Tébessa", "Tlemcen", "Tiaret",
    "Tizi Ouzou", "Alger", "Djelfa", "Jijel", "Sétif", "Saïda", "Skikda",
    "Sidi Bel Abbès", "Annaba", "Guelma", "Constantine", "Médéa", "Mostaganem",
    "M'Sila", "Mascara", "Ouargla", "Oran", "El Bayadh", "Illizi", "Bordj Bou Arréridj",
    "Boumerdès", "El Tarf", "Tindouf", "Tissemsilt", "El Oued", "Khenchela",
    "Souk Ahras", "Tipaza", "Mila", "Aïn Defla", "Naâma", "Aïn Témouchent",
    "Ghardaïa", "Relizane", "Timimoun", "Bordj Badji Mokhtar", "Ouled Djellal",
    "Béni Abbès", "In Salah", "In Guezzam", "Touggourt", "Djanet",
    "El M'Ghair", "El Meniaa",
]  # fmt: skip

# Approximate grouping by wilaya code; it's what drives delivery distance and cost tiers.
HAUTS_PLATEAUX = {3, 4, 5, 12, 14, 17, 19, 20, 28, 32, 34, 38, 40, 45}
SUD = {1, 7, 8, 11, 30, 33, 37, 39, 47, *range(49, 59)}

# Approximate (±1 day, set by moon sighting). Verify against official announcements.
RAMADAN = [
    ("2016-06-06", "2016-07-05"),
    ("2017-05-27", "2017-06-24"),
    ("2018-05-16", "2018-06-14"),
    ("2019-05-06", "2019-06-03"),
    ("2020-04-24", "2020-05-23"),
    ("2021-04-13", "2021-05-12"),
    ("2022-04-02", "2022-05-01"),
    ("2023-03-23", "2023-04-20"),
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-30"),
    ("2026-02-18", "2026-03-19"),
    ("2027-02-08", "2027-03-09"),
]
EIDS = [  # Eid al-Fitr and Eid al-Adha
    "2016-07-06",
    "2016-09-12",
    "2017-06-25",
    "2017-09-01",
    "2018-06-15",
    "2018-08-21",
    "2019-06-04",
    "2019-08-11",
    "2020-05-24",
    "2020-07-31",
    "2021-05-13",
    "2021-07-20",
    "2022-05-02",
    "2022-07-09",
    "2023-04-21",
    "2023-06-28",
    "2024-04-10",
    "2024-06-16",
    "2025-03-30",
    "2025-06-06",
    "2026-03-20",
    "2026-05-27",
    "2027-03-10",
    "2027-05-17",
]
PRE_EID_DAYS = 14


def _key(name: str) -> str:
    ascii_ = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return "".join(ch for ch in ascii_.lower() if ch.isalnum())


_CODE_BY_KEY = {_key(n): i + 1 for i, n in enumerate(WILAYAS)}


def wilaya_code(name: str | None) -> int | None:
    """Accent/case/punctuation-insensitive lookup; None for unknown values like 'Inconnue'."""
    if not isinstance(name, str):
        return None
    return _CODE_BY_KEY.get(_key(name))


def region(code: int | None) -> str:
    if code is None or pd.isna(code):
        return "Inconnue"
    if code in SUD:
        return "Sud"
    if code in HAUTS_PLATEAUX:
        return "Hauts-Plateaux"
    return "Nord"


def to_local(ts: pd.Series) -> pd.Series:
    """Prisma stores naive UTC timestamps; convert to Algiers time (UTC+1, no DST)."""
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert(TZ)


def calendar_features(local: pd.Series) -> pd.DataFrame:
    day = local.dt.tz_localize(None).dt.normalize()
    in_ramadan = np.zeros(len(day), dtype=bool)
    for start, end in RAMADAN:
        in_ramadan |= ((day >= start) & (day <= end)).to_numpy()

    eids = pd.to_datetime(EIDS).to_numpy()
    days_to = (eids[None, :] - day.to_numpy()[:, None]) / np.timedelta64(1, "D")
    days_to = np.where(days_to >= 0, days_to, np.inf).min(axis=1)

    return pd.DataFrame(
        {
            "is_weekend_dz": local.dt.dayofweek.isin([4, 5]).astype(int).to_numpy(),  # Friday, Saturday
            "in_ramadan": in_ramadan.astype(int),
            "pre_eid": ((days_to > 0) & (days_to <= PRE_EID_DAYS)).astype(int),
            "days_to_eid": np.minimum(days_to, 90),
        },
        index=local.index,
    )
