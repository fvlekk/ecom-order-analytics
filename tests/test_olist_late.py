import pandas as pd

from order_analytics.olist_late import build


def order(oid, seller, customer, bought, promised, delivered, status="delivered", route=("SP", "BA")):
    day = lambda d: pd.Timestamp("2018-01-01") + pd.Timedelta(days=d)  # noqa: E731
    return {
        "order_id": oid, "seller_id": seller, "customer_unique_id": customer, "order_status": status,
        "purchased_at": day(bought), "order_estimated_delivery_date": day(promised),
        "order_delivered_customer_date": day(delivered) if delivered is not None else pd.NaT,
        "freight_value": 10.0, "items_value": 100.0, "seller_state": route[0], "customer_state": route[1],
    }  # fmt: skip


def test_route_stats_use_only_recent_completed_deliveries():
    orders = [order(f"old{i}", "s", f"c{i}", bought=0, promised=10, delivered=4) for i in range(20)]  # 4-day transit
    orders += [order(f"new{i}", "s", f"d{i}", bought=90, promised=100, delivered=98) for i in range(20)]  # 8-day
    # Probe orders are undelivered, so they read the route stats without adding to them.
    probe = lambda oid, bought, promised: order(oid, "s", oid, bought, promised, None, status="shipped")  # noqa: E731
    orders += [probe("q1", 50, 60), probe("q2", 80, 90), probe("q3", 98, 110), probe("q4", 99, 103)]
    df = build(pd.DataFrame(orders)).set_index("order_id")

    assert df.loc["q1", "route_recent_transit"] == 4  # day-4 batch is 46 days old: inside the window
    assert pd.isna(df.loc["q2", "route_recent_transit"])  # 76 days old: outside the window
    assert pd.isna(df.loc["q3", "route_recent_transit"])  # day-98 deliveries aren't known on day 98
    assert df.loc["q4", "route_recent_transit"] == 8
    assert df.loc["q4", "promise_slack"] == 4 - 8


def test_label_and_seller_history_use_only_completed_deliveries():
    df = build(pd.DataFrame([
        order("a", "s1", "c1", bought=0, promised=5, delivered=9),   # late, delivered on day 9
        order("b", "s1", "c2", bought=3, promised=8, delivered=6),   # bought before "a" arrived
        order("c", "s1", "c3", bought=10, promised=15, delivered=12),
        order("d", "s2", "c4", bought=11, promised=20, delivered=None, status="shipped"),
    ])).set_index("order_id")  # fmt: skip

    assert df.loc["a", "label"] == 1
    assert df.loc["b", "label"] == 0
    assert pd.isna(df.loc["d", "label"])
    assert list(df.loc[["a", "b", "c"], "seller_prior_deliveries"]) == [0, 0, 2]
    assert list(df.loc[["a", "b", "c"], "seller_prior_late"]) == [0, 0, 1]
    assert df.loc["d", "seller_prior_deliveries"] == 0


def test_orders_after_export_cutoff_are_dropped():
    df = build(
        pd.DataFrame([order("x", "s", "c", bought=0, promised=5, delivered=3)]).assign(
            purchased_at=pd.Timestamp("2018-09-15")
        )
    )
    assert df.empty
