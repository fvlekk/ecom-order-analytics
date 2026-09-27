import numpy as np
import pandas as pd

from order_analytics.monitoring import drift_report, level, psi, rate_shift

rng = np.random.default_rng(0)


def test_psi_is_near_zero_for_the_same_distribution():
    a, b = pd.Series(rng.normal(0, 1, 5000)), pd.Series(rng.normal(0, 1, 5000))
    assert psi(a, b) < 0.02


def test_psi_flags_a_shifted_distribution():
    a, b = pd.Series(rng.normal(0, 1, 5000)), pd.Series(rng.normal(1, 1, 5000))
    assert level(psi(a, b)) == "major"


def test_psi_on_categories_including_new_ones():
    a = pd.Series(["Alger"] * 80 + ["Oran"] * 20)
    b = pd.Series(["Alger"] * 40 + ["Oran"] * 20 + ["Djelfa"] * 40)
    assert psi(a, b) > 0.25


def test_rate_shift_significance():
    assert rate_shift(76, 1000, 35, 1000)["significant"]
    assert not rate_shift(76, 1000, 70, 1000)["significant"]


def test_report_verdict_asks_for_retraining_on_major_drift():
    ref = pd.DataFrame({"x": rng.normal(0, 1, 2000), "y": rng.random(2000) < 0.1})
    cur = pd.DataFrame({"x": rng.normal(2, 1, 2000), "y": rng.random(2000) < 0.1})
    rep = drift_report(ref, cur, ["x"], "y")
    assert rep["major_drift"] == ["x"]
    assert rep["verdict"].startswith("retrain")


def test_psi_handles_pandas_string_dtype():
    a = pd.Series(["Alger"] * 80 + ["Oran"] * 20, dtype="str")
    b = pd.Series(["Alger"] * 20 + ["Oran"] * 80, dtype="str")
    assert level(psi(a, b)) == "major"


def test_small_windows_are_not_judged():
    ref = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.random(500) < 0.1})
    cur = pd.DataFrame({"x": rng.normal(2, 1, 40), "y": rng.random(40) < 0.1})
    assert drift_report(ref, cur, ["x"], "y")["verdict"].startswith("not enough data")
