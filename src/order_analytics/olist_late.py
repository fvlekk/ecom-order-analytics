"""Olist: predict at purchase time whether an order will arrive after its promised date.

Data: Olist Brazilian E-Commerce Public Dataset, CC BY-NC-SA 4.0. Non-commercial use only,
so models trained here must not ship in Bizz; only the method is reused.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from order_analytics.config import OLIST_ORDERS, OLIST_REPORTS, PROCESSED_DIR, SEED
from order_analytics.rolling import prior_window_sums
from order_analytics.train import evaluate

# September-October 2018 hold 20 orders and no deliveries: the export ends in August.
LAST_PURCHASE = pd.Timestamp("2018-08-31 23:59:59")

BASE_NUMERIC = [
    "promised_days", "distance_km", "same_state", "freight_value", "items_value", "freight_ratio",
    "n_items", "n_sellers", "weight_g", "payment_installments", "n_payments",
    "seller_prior_deliveries", "seller_prior_late", "seller_prior_late_rate",
    "customer_prior_deliveries", "customer_prior_late",
    "orders_last_7d", "month", "dayofweek", "hour",
]  # fmt: skip
# `month` lets a model memorize past crises (Feb-Mar 2018); recent route stats adapt instead.
ROUTE_NUMERIC = ["route_recent_transit", "route_recent_late_rate", "route_recent_n", "promise_slack"]
CATEGORICAL = ["customer_state", "seller_state", "payment_type", "main_category"]
# Only signals that are recomputed from recent history, so they can't encode a stale regime.
ADAPTIVE_NUMERIC = [
    "promise_slack", "promised_days", "route_recent_transit", "route_recent_late_rate", "route_recent_n",
    "orders_last_7d", "payment_installments", "n_items", "weight_g",
]  # fmt: skip
FEATURE_SETS = {
    "static": (BASE_NUMERIC, CATEGORICAL),
    "drift_aware": ([f for f in BASE_NUMERIC if f != "month"] + ROUTE_NUMERIC, CATEGORICAL),
    "adaptive": (ADAPTIVE_NUMERIC, ["payment_type"]),
}
TEST_MONTHS = ["2018-06", "2018-07", "2018-08"]
RECENT_WINDOW = "60D"
MIN_ROUTE_DELIVERIES = 20
ATTRIBUTION = "Data: Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0)"


def label(df: pd.DataFrame) -> pd.Series:
    """1 = delivered after the promised day, 0 = on time, NaN = never delivered (not in this task)."""
    delivered = df["order_status"].eq("delivered") & df["order_delivered_customer_date"].notna()
    late = df["order_delivered_customer_date"].dt.normalize() > df["order_estimated_delivery_date"]
    return late.astype(float).where(delivered)


def prior_history(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """For each order: deliveries (and late ones) for `key` that were completed before the purchase."""
    events = df.loc[df["label"].notna() & df[key].notna(), [key, "order_delivered_customer_date", "label"]]
    events = events.sort_values("order_delivered_customer_date").rename(columns={"order_delivered_customer_date": "t"})
    events["n"] = events.groupby(key).cumcount() + 1
    events["late"] = events.groupby(key)["label"].cumsum()

    left = df.loc[df[key].notna(), ["order_id", key, "purchased_at"]].sort_values("purchased_at")
    merged = pd.merge_asof(
        left, events[[key, "t", "n", "late"]], left_on="purchased_at", right_on="t",
        by=key, direction="backward", allow_exact_matches=False,
    )  # fmt: skip
    return merged.set_index("order_id")[["n", "late"]].reindex(df["order_id"]).fillna(0).set_axis(df.index)


def recent_stats(df: pd.DataFrame, keys: list[str], window: str = RECENT_WINDOW) -> pd.DataFrame:
    """Deliveries completed on the same `keys` in the `window` before the purchase day (strictly)."""
    d = df.assign(
        transit=(df["order_delivered_customer_date"] - df["purchased_at"]).dt.total_seconds() / 86400,
        late=df["label"],
    )
    sums = prior_window_sums(
        d, keys, query_time="purchased_at", event_time="order_delivered_customer_date",
        values=["transit", "late"], window=window, known=df["label"].notna(),
    )  # fmt: skip
    n = sums["n"].replace(0, np.nan)
    return pd.DataFrame({"n": sums["n"], "transit": sums["transit"] / n, "late_rate": sums["late"] / n})


def build(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders[orders["purchased_at"] <= LAST_PURCHASE].sort_values("purchased_at", ignore_index=True).copy()
    df["label"] = label(df)

    df["promised_days"] = (df["order_estimated_delivery_date"] - df["purchased_at"]).dt.total_seconds() / 86400
    df["freight_ratio"] = df["freight_value"] / df["items_value"].replace(0, np.nan)
    df["month"] = df["purchased_at"].dt.month
    df["dayofweek"] = df["purchased_at"].dt.dayofweek
    df["hour"] = df["purchased_at"].dt.hour
    # Platform load: orders placed in the previous 7 days (Black Friday-style peaks strain logistics).
    counts = pd.Series(1, index=df["purchased_at"]).rolling("7D", closed="left").count()
    df["orders_last_7d"] = counts.to_numpy()

    for key, prefix in [("seller_id", "seller"), ("customer_unique_id", "customer")]:
        hist = prior_history(df, key)
        df[f"{prefix}_prior_deliveries"] = hist["n"]
        df[f"{prefix}_prior_late"] = hist["late"]
    df["seller_prior_late_rate"] = df["seller_prior_late"] / df["seller_prior_deliveries"].replace(0, np.nan)

    # Recent transit time on this route (seller state -> customer state); thin routes fall back
    # to all deliveries into the customer's state.
    route = recent_stats(df, ["seller_state", "customer_state"])
    dest = recent_stats(df, ["customer_state"])
    thin = route["n"].fillna(0) < MIN_ROUTE_DELIVERIES
    best = route.where(~thin, dest)
    df["route_recent_n"] = best["n"].fillna(0)
    df["route_recent_transit"] = best["transit"]
    df["route_recent_late_rate"] = best["late_rate"]
    df["promise_slack"] = df["promised_days"] - df["route_recent_transit"]
    return df


def models(numeric: list[str], categorical: list[str]) -> dict:
    logreg = make_pipeline(
        ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), categorical),
        ]),
        LogisticRegression(max_iter=3000, C=0.5),
    )  # fmt: skip
    hgb = HistGradientBoostingClassifier(
        categorical_features="from_dtype", learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=SEED,
    )  # fmt: skip
    return {"logistic_regression": logreg, "gradient_boosting": hgb}


def walk_forward(data: pd.DataFrame, month: str) -> tuple[pd.DataFrame, dict, dict]:
    """Train as of the 1st of `month` and score that month's orders, as a monthly retrain would."""
    start = pd.Timestamp(month)
    test = data[(data["purchased_at"] >= start) & (data["purchased_at"] < start + pd.offsets.MonthBegin(1))]
    # On the cutoff day, an order's outcome is known only once its promised date has passed.
    known = data[data["order_estimated_delivery_date"] < start]
    y = known["label"].to_numpy().astype(int)

    state_rate = known.groupby("customer_state", observed=True)["label"].mean()
    scores = {
        "base_rate": np.full(len(test), y.mean()),
        "state_rule": test["customer_state"].map(state_rate).astype(float).fillna(y.mean()).to_numpy(),
        # Single-feature rules: a shorter promise, or less slack vs. the recent route transit, is riskier.
        "short_promise_rule": -test["promised_days"].to_numpy(),
        "slack_rule": -test["promise_slack"].fillna(test["promise_slack"].median()).to_numpy(),
    }
    fitted = {}
    for set_name, (numeric, categorical) in FEATURE_SETS.items():
        cols = numeric + categorical
        for model_name, model in models(numeric, categorical).items():
            name = f"{model_name}[{set_name}]"
            fitted[name] = model.fit(known[cols], y)
            scores[name] = model.predict_proba(test[cols])[:, 1]
    return test, scores, fitted


def main() -> None:
    df = build(pd.read_parquet(OLIST_ORDERS))
    data = df[df["label"].notna()].copy()
    for col in CATEGORICAL:
        data[col] = data[col].astype("category")

    per_month, tests, all_scores = {}, [], {}
    for month in TEST_MONTHS:
        test, scores, fitted = walk_forward(data, month)
        y = test["label"].to_numpy().astype(int)
        per_month[month] = {"orders": len(test), "late_rate": round(float(y.mean()), 4)} | {
            name: evaluate(y, s) for name, s in scores.items()
        }
        tests.append(test)
        for name, s in scores.items():
            all_scores.setdefault(name, []).append(s)

    names = list(all_scores)
    mean = {
        n: {
            m: round(float(np.mean([per_month[mo][n][m] for mo in TEST_MONTHS])), 4)
            for m in per_month[TEST_MONTHS[0]][n]
        }
        for n in names
    }
    ranking = sorted(names, key=lambda n: mean[n]["pr_auc"], reverse=True)
    best_model = next(n for n in ranking if "[" in n)

    # Feature importance of the best model, on the last month (its own out-of-time data).
    numeric, categorical = FEATURE_SETS[best_model.split("[")[1].rstrip("]")]
    cols = numeric + categorical
    imp = permutation_importance(
        fitted[best_model], test[cols], test["label"].astype(int),
        scoring="average_precision", n_repeats=5, random_state=SEED,
    )  # fmt: skip
    importance = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)

    results = {
        "evaluation": "walk-forward: retrain on the 1st of each month on orders whose promised date has passed",
        "test_months": TEST_MONTHS,
        "mean_over_months": {n: mean[n] for n in ranking},
        "per_month": per_month,
        "best_model": best_model,
        "best_model_top_features_last_month": importance.head(10).round(4).to_dict(),
        "attribution": ATTRIBUTION,
    }
    OLIST_REPORTS.mkdir(parents=True, exist_ok=True)
    (OLIST_REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {k: results[k] for k in ["mean_over_months", "best_model", "best_model_top_features_last_month"]}, indent=2
        )
    )

    pooled = pd.concat(tests)
    y_all = pooled["label"].to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for name in ["state_rule", "slack_rule", best_model, "logistic_regression[static]"]:
        s = np.concatenate(all_scores[name])
        prec, rec, _ = precision_recall_curve(y_all, s)
        ax.plot(rec, prec, label=f"{name} (PR-AUC {average_precision_score(y_all, s):.3f})")
    ax.axhline(y_all.mean(), ls="--", c="grey", lw=1, label="base rate")
    ax.set(
        xlabel="Recall (share of late orders caught)",
        ylabel="Precision",
        title="Late-delivery detection, Jun-Aug 2018 (monthly retrain)",
    )
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, ATTRIBUTION, fontsize=6, color="grey")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OLIST_REPORTS / "pr_curve.png", dpi=150)

    keep = ["order_id", "purchased_at", "customer_state", "seller_state", "payment_type", "main_category",
            "promised_days", "promise_slack", "route_recent_transit", "distance_km", "items_value",
            "freight_value", "label"]  # fmt: skip
    out = pooled[keep].copy()
    out["slack_rule_score"] = np.concatenate(all_scores["slack_rule"])
    out["model_score"] = np.concatenate(all_scores[best_model])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED_DIR / "olist_scores_jun_aug_2018.parquet", index=False)


if __name__ == "__main__":
    main()
