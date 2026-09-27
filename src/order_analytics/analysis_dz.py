"""Algerian orders (Bizz): where and when orders fail, with a significance test next to every claim.

Reads data/processed/features.parquet (run `features real` first). Results on real data stay local:
reports/dz/ is gitignored. With ~640 finished orders, only large effects can be told apart from noise.
"""

import json

import pandas as pd

from order_analytics.config import FEATURES, ROOT
from order_analytics.markets import dz
from order_analytics.monitoring import rate_shift

REPORTS = ROOT / "reports" / "dz"
DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MIN_GROUP = 30  # below this, a group's rate is shown but never compared
SWAP_DAYS = 14


def likely_swaps(df: pd.DataFrame, days: int = SWAP_DAYS) -> pd.Series:
    """Failed orders followed by a new order from the same client within `days`.

    The shop records most exchanges as a return plus a new order (the exchange flow is rarely used),
    so these are more likely swaps than lost sales. Uses hindsight: an analysis label, never a feature.
    """
    ordered = df.sort_values("created_at")
    nxt = ordered.groupby("client_id")["created_at"].shift(-1)
    soon = (nxt - ordered["created_at"]).dt.total_seconds() / 86400 <= days
    return (ordered["label"].eq(1) & soon).reindex(df.index)


def fail_table(finished: pd.DataFrame, by: str) -> pd.DataFrame:
    """Orders, failures and failure rate per group, largest groups first."""
    g = finished.groupby(by, dropna=False)["label"].agg(orders="size", failed="sum")
    g["fail_rate"] = (g["failed"] / g["orders"]).round(3)
    return g.sort_values("orders", ascending=False)


def compare(finished: pd.DataFrame, mask: pd.Series, name_in: str, name_out: str, label: str = "label") -> dict:
    """Failure rate inside vs outside `mask`, with a two-proportion z-test (|z| >= 1.96 ~ p < 0.05)."""
    a, b = finished.loc[mask, label], finished.loc[~mask, label]
    out = {"groups": [name_in, name_out], "n": [len(a), len(b)], "failed": [int(a.sum()), int(b.sum())]}
    if min(len(a), len(b)) < MIN_GROUP:
        return {**out, "verdict": f"too few orders to compare (min {MIN_GROUP} per group)"}
    t = rate_shift(int(b.sum()), len(b), int(a.sum()), len(a))
    return {
        **out,
        "fail_rate": [t["current_rate"], t["reference_rate"]],
        "z": t["z"],
        "verdict": "significant" if t["significant"] else "not significant",
    }


def analyse(df: pd.DataFrame) -> dict:
    finished = df[df["label"] >= 0].copy()
    local = dz.to_local(df["created_at"])
    df = df.assign(month=local.dt.strftime("%Y-%m"), weekday=local.dt.dayofweek)
    finished = finished.join(df[["month", "weekday"]])
    finished["swap"] = likely_swaps(df).reindex(finished.index)
    finished["failed_excl_swaps"] = (finished["label"].eq(1) & ~finished["swap"]).astype(int)
    repeat = finished.prior_orders.gt(0)
    entered = finished[finished.source.eq("BIZZ")]

    first, last = local.min().tz_localize(None), local.max().tz_localize(None)
    eids = [e for e in dz.EIDS if first <= pd.Timestamp(e) <= last]
    return {
        "data": {
            "orders": len(df),
            "finished": len(finished),
            "failed": int(finished.label.sum()),
            "fail_rate": round(finished.label.mean(), 3),
            "period": [str(local.min().date()), str(local.max().date())],
            "eids_in_period": eids,
            "likely_swaps": int(finished.swap.sum()),
            "swap_share_of_failures": {
                "repeat client": round(finished.loc[repeat & finished.label.eq(1), "swap"].mean(), 3),
                "first order": round(finished.loc[~repeat & finished.label.eq(1), "swap"].mean(), 3),
            },
        },
        "by_month": fail_table(finished, "month").sort_index().reset_index().to_dict("records"),
        "orders_by_weekday": {
            src: {DAYS[d]: int(n) for d, n in g["weekday"].value_counts().sort_index().items()}
            for src, g in df.groupby("source")
        },
        "by_region": fail_table(finished, "region").reset_index().to_dict("records"),
        "by_carrier": fail_table(finished, "carrier").reset_index().to_dict("records"),
        "tests": {
            # The livreur field is rarely filled; no Yalidine tracking means a private livreur (Alger).
            "yalidine_vs_private": compare(
                finished, finished.has_yalidine_tracking.eq(1), "Yalidine", "private livreur"
            ),
            "imported_vs_entered": compare(finished, finished.source.eq("YALIDINE"), "imported", "entered in Bizz"),
            "pre_eid_vs_rest": compare(
                finished, finished.pre_eid.eq(1), f"{dz.PRE_EID_DAYS} days before an Eid", "rest"
            ),
            "pre_eid_entered_only": compare(
                entered, entered.pre_eid.eq(1), "pre-Eid (entered in Bizz)", "rest (entered in Bizz)"
            ),
            "repeat_vs_first": compare(finished, finished.prior_orders.gt(0), "repeat client", "first order"),
            # Imports go almost only to a few loyal recipients, so repeat and import effects overlap.
            "repeat_vs_first_entered_only": compare(
                entered, entered.prior_orders.gt(0), "repeat client (entered)", "first order (entered)"
            ),
            # Same questions once likely swaps (return + reorder within SWAP_DAYS) are not counted as failures.
            "repeat_vs_first_swaps_excluded": compare(
                finished, repeat, "repeat client", "first order", label="failed_excl_swaps"
            ),
            "imported_vs_entered_swaps_excluded": compare(
                finished, finished.source.eq("YALIDINE"), "imported", "entered in Bizz", label="failed_excl_swaps"
            ),
            "alger_vs_rest": compare(finished, finished.wilaya.eq("Alger"), "Alger", "other wilayas"),
            "weekend_vs_week": compare(finished, finished.is_weekend_dz.eq(1), "Fri-Sat", "Sun-Thu"),
        },
    }


def main() -> None:
    report = analyse(pd.read_parquet(FEATURES))
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    d = report["data"]
    print(f"{d['orders']} orders, {d['finished']} finished, {d['failed']} failed ({d['fail_rate']:.1%}), {d['period']}")
    for name, t in report["tests"].items():
        rates = " vs ".join(f"{r:.1%}" for r in t.get("fail_rate", [])) or "-"
        print(f"  {name:22} n={t['n']}  {rates}  z={t.get('z', '-')}  {t['verdict']}")
    print(f"-> {REPORTS / 'analysis.json'}")


if __name__ == "__main__":
    main()
