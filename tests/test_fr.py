import pandas as pd
import pytest

from order_analytics.markets import fr


def test_13_regions_96_departments():
    assert len(fr.REGIONS) == 13
    assert len(fr.REGION_BY_DEPT) == 96


@pytest.mark.parametrize(
    ("cp", "dept", "reg"),
    [("75011", "75", "Île-de-France"), ("13001", "13", "Provence-Alpes-Côte d'Azur"),
     ("20000", "2A", "Corse"), ("20200", "2B", "Corse"), ("97400", None, "Inconnue"), ("7501", None, "Inconnue")],
)  # fmt: skip
def test_postcode_to_department_and_region(cp, dept, reg):
    assert fr.department(cp) == dept
    assert fr.region(fr.department(cp)) == reg


def test_easter_and_holidays_2026():
    assert fr.easter(2026) == pd.Timestamp("2026-04-05").date()
    h = fr.public_holidays(2026)
    for d in ["2026-04-06", "2026-05-14", "2026-05-25", "2026-07-14"]:  # lundi de Pâques, Ascension, Pentecôte
        assert pd.Timestamp(d).date() in h


def test_sales_and_black_friday_2026():
    (winter_start, _), (summer_start, _) = fr.sales_periods(2026)
    assert str(winter_start) == "2026-01-14"  # 2nd Wednesday of January
    assert str(summer_start) == "2026-06-24"  # last Wednesday of June
    assert str(fr.black_friday(2026)) == "2026-11-27"


def test_paris_time_follows_dst():
    ts = pd.Series(pd.to_datetime(["2026-01-15 12:00", "2026-07-15 12:00"]))  # naive UTC
    assert list(fr.to_local(ts).dt.hour) == [13, 14]  # UTC+1 in winter, UTC+2 in summer
