"""Drift monitoring: has the world moved since the model was trained?

Compares a reference window with a recent window. For each feature, the Population Stability
Index (PSI); for the outcome, a two-proportion z-test. Conventional PSI reading: < 0.10 stable,
0.10-0.25 watch, > 0.25 major shift. A model is only as good as the regime it learned.
"""

import json
import math
import sys

import numpy as np
import pandas as pd

from order_analytics.config import ROOT

EPS = 1e-4
MIN_ROWS = 100  # PSI on a few dozen rows mostly measures noise
REPORTS = ROOT / "reports" / "monitoring"


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref, cur = reference.dropna(), current.dropna()
    if ref.empty or cur.empty:
        return float("nan")
    if not pd.api.types.is_numeric_dtype(ref) or ref.nunique() <= bins:
        cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        e = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(ref, edges)[0] / len(ref)
        a = np.histogram(cur, edges)[0] / len(cur)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def level(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return "stable" if value < 0.10 else "watch" if value < 0.25 else "major"


def rate_shift(ref_pos: int, ref_n: int, cur_pos: int, cur_n: int) -> dict:
    p1, p2 = ref_pos / ref_n, cur_pos / cur_n
    pooled = (ref_pos + cur_pos) / (ref_n + cur_n)
    se = math.sqrt(pooled * (1 - pooled) * (1 / ref_n + 1 / cur_n)) or float("nan")
    z = (p2 - p1) / se
    return {
        "reference_rate": round(p1, 4),
        "current_rate": round(p2, 4),
        "z": round(z, 2),
        "significant": abs(z) >= 1.96,
    }


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], label: str) -> dict:
    rows = [{"feature": f, "psi": round(psi(ref[f], cur[f]), 3)} for f in features]
    for r in rows:
        r["level"] = level(r["psi"])
    rows.sort(key=lambda r: -1 if math.isnan(r["psi"]) else -r["psi"])
    y_ref, y_cur = ref[label].dropna(), cur[label].dropna()
    outcome = rate_shift(int(y_ref.sum()), len(y_ref), int(y_cur.sum()), len(y_cur))
    majors = [r["feature"] for r in rows if r["level"] == "major"]
    if min(len(ref), len(cur)) < MIN_ROWS:
        verdict = f"not enough data to judge (a window has fewer than {MIN_ROWS} finished orders)"
    elif majors or outcome["significant"]:
        verdict = "retrain on recent data, and fall back to the simple rule until the new model beats it"
    elif any(r["level"] == "watch" for r in rows):
        verdict = "watch: re-check next period"
    else:
        verdict = "stable"
    return {
        "reference": {"rows": len(ref)}, "current": {"rows": len(cur)},
        "outcome_shift": outcome, "features": rows, "major_drift": majors, "verdict": verdict,
    }  # fmt: skip


def _olist() -> dict:
    from order_analytics.config import OLIST_HINT, OLIST_ORDERS, require
    from order_analytics.olist_late import ADAPTIVE_NUMERIC, build

    df = build(pd.read_parquet(require(OLIST_ORDERS, OLIST_HINT)))
    df = df[df["label"].notna()]
    ref = df[(df["purchased_at"] >= "2018-01-01") & (df["purchased_at"] < "2018-05-01")]
    cur = df[(df["purchased_at"] >= "2018-06-01") & (df["purchased_at"] < "2018-09-01")]
    feats = ["distance_km", "freight_value", "seller_prior_late_rate", "customer_state", *ADAPTIVE_NUMERIC]
    report = drift_report(ref, cur, list(dict.fromkeys(feats)), "label")
    report["windows"] = {"reference": "2018-01 to 2018-04", "current": "2018-06 to 2018-08"}
    return report


def _bizz() -> dict:
    from order_analytics.config import PRIVATE_HINT, RAW_ORDERS, require
    from order_analytics.features import build

    df = build(pd.read_parquet(require(RAW_ORDERS, PRIVATE_HINT)))
    df = df[df["label"] >= 0]
    cut = df["created_at"].max() - pd.Timedelta(days=60)
    ref, cur = df[df["created_at"] < cut], df[df["created_at"] >= cut]
    feats = [
        "wilaya",
        "carrier",
        "source",
        "items_value",
        "n_items",
        "prior_orders",
        "route_recent_fail_rate",
        "dayofweek",
    ]
    report = drift_report(ref, cur, feats, "label")
    report["windows"] = {"reference": f"before {cut.date()}", "current": f"last 60 days (from {cut.date()})"}
    return report


def main(which: str = "olist") -> None:
    report = {"olist": _olist, "bizz": _bizz}[which]()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{which}.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ["windows", "outcome_shift", "major_drift", "verdict"]}, indent=2))
    for r in report["features"][:8]:
        print(f"  {r['feature']:26s} PSI {r['psi']:>7}  {r['level']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "olist")
