import pandas as pd
import pytest

from order_analytics.features import build, trust_level


def order(client, created_day, resolved_day, statut):
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    return {
        "tenant_id": "t1",
        "client_id": client,
        "livreur_id": None,
        "statut": statut,
        "source": "BIZZ",
        "delivery_fee": 500.0,
        "fee_paid_by": "CLIENT",
        "credit_discount": 0.0,
        "promo_discount": 0.0,
        "from_live": False,
        "is_exchange_order": False,
        "has_yalidine_tracking": True,
        "created_at": t0 + pd.Timedelta(days=created_day),
        "updated_at": t0 + pd.Timedelta(days=resolved_day),
        "wilaya": "Alger",
        "client_created_at": t0,
        "n_items": 1,
        "items_value": 3000.0,
        "items_cost": 1500.0,
    }


def test_history_only_counts_outcomes_known_at_order_time():
    df = (
        build(
            pd.DataFrame(
                [
                    order("c1", 0, 5, "RETOUR"),
                    order("c1", 2, 3, "LIVREE"),  # placed before the first order was resolved
                    order("c1", 6, 7, "LIVREE"),
                    order("c2", 1, 2, "ANNULEE"),  # another client must not leak into c1
                ]
            )
        )
        .set_index(["client_id", "created_at"])
        .sort_index()
    )

    rows = df.loc["c1"]
    assert list(rows["prior_orders"]) == [0, 1, 2]
    assert list(rows["prior_failed"]) == [0, 0, 1]
    assert list(rows["prior_ok"]) == [0, 0, 1]
    assert list(rows["trust_level"]) == ["nouveau", "nouveau", "prudence"]


def test_labels():
    df = build(
        pd.DataFrame(
            [
                order("c1", 0, 1, "RECUPERE"),
                order("c2", 0, 1, "AVEC_ECHANGE"),
                order("c3", 0, 1, "EN_LIVRAISON"),
            ]
        )
    ).set_index("client_id")
    assert df.loc["c1", "label"] == 1
    assert df.loc["c2", "label"] == 0
    assert df.loc["c3", "label"] == -1


@pytest.mark.parametrize(
    ("a", "r", "l", "expected"),
    [
        (2, 0, 5, "risque"),
        (0, 2, 0, "risque"),
        (1, 2, 0, "risque"),
        (1, 1, 9, "prudence"),
        (0, 0, 2, "fiable"),
        (0, 0, 1, "nouveau"),
    ],
)
def test_trust_rule_matches_bizz(a, r, l, expected):
    assert trust_level(a, r, l) == expected


def test_route_failure_rate_only_uses_outcomes_known_before_the_order_day():
    # Five failed then resolved orders to Alger via Yalidine (no livreur, tracking -> carrier Yalidine).
    past = [order(f"p{i}", 0, 2, "RETOUR") for i in range(5)]
    probe_same_day = order("q1", 2, 9, "EN_LIVRAISON")  # resolutions on day 2 aren't known on day 2
    probe_next_day = order("q2", 3, 9, "EN_LIVRAISON")
    df = build(pd.DataFrame([*past, probe_same_day, probe_next_day])).set_index("client_id")
    assert df.loc["q1", "route_recent_n"] == 0
    assert df.loc["q2", "route_recent_n"] == 5
    assert df.loc["q2", "route_recent_fail_rate"] == 1.0
    assert df.loc["q2", "carrier"] == "Yalidine"
