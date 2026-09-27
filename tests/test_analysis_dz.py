import pandas as pd

from order_analytics.analysis_dz import compare, fail_table, likely_swaps


def _orders(n_in, failed_in, n_out, failed_out):
    label = [1] * failed_in + [0] * (n_in - failed_in) + [1] * failed_out + [0] * (n_out - failed_out)
    return pd.DataFrame({"label": label, "group": ["a"] * n_in + ["b"] * n_out})


def test_fail_table_counts_and_rates():
    t = fail_table(_orders(40, 10, 60, 6), "group")
    assert t.loc["a", "failed"] == 10 and t.loc["b", "orders"] == 60
    assert t.loc["a", "fail_rate"] == 0.25


def test_compare_flags_a_clear_gap_and_skips_small_groups():
    df = _orders(200, 60, 200, 20)
    r = compare(df, df.group.eq("a"), "a", "b")
    assert r["verdict"] == "significant" and r["z"] > 0 and r["fail_rate"] == [0.3, 0.1]
    small = _orders(10, 5, 200, 20)
    assert compare(small, small.group.eq("a"), "a", "b")["verdict"].startswith("too few")


def test_likely_swaps_need_a_reorder_by_the_same_client_soon_after():
    t = pd.to_datetime(["2026-05-01", "2026-05-05", "2026-05-01", "2026-06-20", "2026-05-02"])
    df = pd.DataFrame({"client_id": ["a", "a", "b", "b", "c"], "created_at": t, "label": [1, 0, 1, 0, 1]})
    # a: failed then reordered after 4 days -> swap; b: reordered after 50 days -> not; c: never reordered
    assert likely_swaps(df).tolist() == [True, False, False, False, False]
