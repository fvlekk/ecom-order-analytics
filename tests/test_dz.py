import pandas as pd
import pytest

from order_analytics.markets import dz


def test_58_wilayas_with_official_codes():
    assert len(dz.WILAYAS) == 58
    assert dz.wilaya_code("Alger") == 16
    assert dz.wilaya_code("Oran") == 31
    assert dz.wilaya_code("El Meniaa") == 58


@pytest.mark.parametrize("raw", ["Béjaïa", "bejaia", "BEJAIA", " Béjaïa "])
def test_wilaya_lookup_ignores_accents_case_spacing(raw):
    assert dz.wilaya_code(raw) == 6


@pytest.mark.parametrize("raw", ["Inconnue", "", None])
def test_unknown_wilaya(raw):
    assert dz.wilaya_code(raw) is None
    assert dz.region(dz.wilaya_code(raw)) == "Inconnue"


def test_regions():
    assert dz.region(dz.wilaya_code("Alger")) == "Nord"
    assert dz.region(dz.wilaya_code("Sétif")) == "Hauts-Plateaux"
    assert dz.region(dz.wilaya_code("Tamanrasset")) == "Sud"
    assert {dz.region(c) for c in range(1, 59)} == {"Nord", "Hauts-Plateaux", "Sud"}


def test_naive_utc_becomes_algiers_time():
    # Thursday 23:30 UTC is already Friday 00:30 in Algiers, so it's the weekend.
    local = dz.to_local(pd.Series(pd.to_datetime(["2026-05-21 23:30"])))
    assert local.dt.hour.iloc[0] == 0
    assert dz.calendar_features(local)["is_weekend_dz"].iloc[0] == 1


def test_ramadan_and_pre_eid_windows():
    days = ["2026-03-01 12:00", "2026-03-15 12:00", "2026-05-20 12:00", "2026-07-01 12:00"]
    cal = dz.calendar_features(dz.to_local(pd.Series(pd.to_datetime(days))))
    assert list(cal["in_ramadan"]) == [1, 1, 0, 0]
    # Eid al-Fitr 2026-03-20 and Eid al-Adha 2026-05-27; the window is the 14 days before.
    assert list(cal["pre_eid"]) == [0, 1, 1, 0]
    assert list(cal["days_to_eid"]) == [19, 5, 7, 90]
