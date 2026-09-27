"""Order-level features using only information available when the order was placed."""

import numpy as np
import pandas as pd

from order_analytics.config import FAILED, FEATURES, PROCESSED_DIR, SUCCEEDED
from order_analytics.markets import dz
from order_analytics.rolling import prior_window_sums

TRUST_LEVELS = ["nouveau", "fiable", "prudence", "risque"]
ROUTE_WINDOW = "60D"
MIN_ROUTE_ORDERS = 5  # small shops: below this, fall back to the wilaya alone


def trust_level(annulees: int, retours: int, livrees: int) -> str:
    """Same rule as bizz/server/src/lib/trust.js computeTrust."""
    if annulees >= 2 or retours >= 2 or annulees + retours >= 3:
        return "risque"
    if annulees >= 1 or retours >= 1:
        return "prudence"
    if livrees >= 2:
        return "fiable"
    return "nouveau"


def _client_history(g: pd.DataFrame) -> pd.DataFrame:
    created = g["created_at"].to_numpy()
    resolved = g["updated_at"].to_numpy()
    statut = g["statut"].to_numpy()
    out = np.zeros((len(g), 6))
    for i in range(len(g)):
        earlier = created < created[i]
        # An earlier order's outcome only counts once it was known (last update before this order).
        known = earlier & (resolved <= created[i])
        s = statut[known]
        out[i] = [
            earlier.sum(),
            np.isin(s, list(SUCCEEDED)).sum(),
            np.isin(s, list(FAILED)).sum(),
            (s == "ANNULEE").sum(),
            (s == "RETOUR").sum(),
            (s == "LIVREE").sum(),
        ]
    hist = pd.DataFrame(
        out,
        index=g.index,
        columns=["prior_orders", "prior_ok", "prior_failed", "prior_annulee", "prior_retour", "prior_livree"],
    ).astype(int)
    hist["days_since_prev_order"] = g["created_at"].diff().dt.total_seconds() / 86400
    return hist


def build(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.sort_values("created_at", ignore_index=True).copy()

    hist = df.groupby("client_id", group_keys=False)[["created_at", "updated_at", "statut"]].apply(_client_history)
    df = df.join(hist)

    known = df["prior_ok"] + df["prior_failed"]
    df["prior_fail_rate"] = (df["prior_failed"] + 1) / (known + 2)
    df["trust_level"] = [
        trust_level(a, r, l)
        for a, r, l in zip(df["prior_annulee"], df["prior_retour"], df["prior_livree"], strict=True)
    ]
    df["client_tenure_days"] = (df["created_at"] - df["client_created_at"]).dt.total_seconds() / 86400

    df["fee_paid_by_client"] = (df["fee_paid_by"] == "CLIENT").astype(int)
    df["has_livreur"] = df["livreur_id"].notna().astype(int)
    # Yalidine imports hardcode prixAchat=0, so a zero cost means "unknown", not a 100% margin.
    cost = df["items_cost"].replace(0, np.nan)
    df["margin_ratio"] = (df["items_value"] - cost) / df["items_value"].replace(0, np.nan)

    local = dz.to_local(df["created_at"])
    df["hour"] = local.dt.hour
    df["dayofweek"] = local.dt.dayofweek
    df = df.join(dz.calendar_features(local))
    df["wilaya_code"] = df["wilaya"].map(dz.wilaya_code).astype("Int64")
    df["region"] = df["wilaya_code"].map(dz.region, na_action=None)

    for col in ["from_live", "is_exchange_order", "has_yalidine_tracking"]:
        df[col] = df[col].astype(int)

    df["label"] = np.select([df["statut"].isin(FAILED), df["statut"].isin(SUCCEEDED)], [1, 0], default=-1)

    # Same idea as Olist's promise slack: recent evidence on this route (wilaya x carrier), using only
    # orders whose outcome was known (last update) before this order's day. Recomputed daily, so it
    # follows drift instead of freezing a past regime.
    df["carrier"] = np.select(
        [df["has_yalidine_tracking"] == 1, df["has_livreur"] == 1], ["Yalidine", "Livreur"], "Autre"
    )
    known = df["label"] >= 0
    d = df.assign(failed=(df["label"] == 1).astype(float))
    route = prior_window_sums(d, ["wilaya", "carrier"], "created_at", "updated_at", ["failed"], ROUTE_WINDOW, known)
    dest = prior_window_sums(d, ["wilaya"], "created_at", "updated_at", ["failed"], ROUTE_WINDOW, known)
    best = route.where(route["n"] >= MIN_ROUTE_ORDERS, dest)
    df["route_recent_n"] = best["n"].fillna(0)
    df["route_recent_fail_rate"] = best["failed"] / best["n"].replace(0, np.nan)
    return df


def main(source: str = "synthetic") -> None:
    from order_analytics.config import RAW_ORDERS, SYNTHETIC_ORDERS

    path = SYNTHETIC_ORDERS if source == "synthetic" else RAW_ORDERS
    df = build(pd.read_parquet(path))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES, index=False)
    print(f"{len(df)} rows, {(df['label'] >= 0).sum()} labelled -> {FEATURES}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
