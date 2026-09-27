"""Time-split evaluation: Bizz's trust rule vs. logistic regression vs. gradient boosting."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from order_analytics.config import FEATURES, FIGURES_DIR, PROCESSED_DIR, ROOT, SEED

NUMERIC = [
    "prior_orders",
    "prior_ok",
    "prior_failed",
    "prior_fail_rate",
    "days_since_prev_order",
    "client_tenure_days",
    "delivery_fee",
    "fee_paid_by_client",
    "credit_discount",
    "promo_discount",
    "from_live",
    "is_exchange_order",
    "has_yalidine_tracking",
    "has_livreur",
    "n_items",
    "items_value",
    "margin_ratio",
    "hour",
    "dayofweek",
    "is_weekend_dz",
    "in_ramadan",
    "pre_eid",
    "days_to_eid",
    "route_recent_n",
    "route_recent_fail_rate",
]
CATEGORICAL = ["wilaya", "region", "tenant_id", "trust_level", "source", "carrier"]
# Higher = riskier, so the rule can be scored with the same ranking metrics as the models.
TRUST_SCORE = {"fiable": 0, "nouveau": 1, "prudence": 2, "risque": 3}
REVIEW_SHARE = 0.20


def time_split(
    df: pd.DataFrame, test_share: float = 0.2, time_col: str = "created_at"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(time_col, ignore_index=True)
    cut = int(len(df) * (1 - test_share))
    return df.iloc[:cut], df.iloc[cut:]


def recall_at_share(y: np.ndarray, score: np.ndarray, share: float) -> float:
    """Share of all failed orders caught if the shop double-checks the top `share` riskiest orders."""
    k = max(1, int(len(y) * share))
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].sum() / max(1, y.sum()))


def evaluate(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "pr_auc": round(float(average_precision_score(y, score)), 4),
        f"recall_at_{int(REVIEW_SHARE * 100)}pct": round(recall_at_share(y, score, REVIEW_SHARE), 4),
    }


def models() -> dict:
    logreg = make_pipeline(
        ColumnTransformer(
            [
                ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAL),
            ]
        ),
        LogisticRegression(max_iter=2000, C=0.5),
    )
    hgb = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=SEED,
    )
    return {"logistic_regression": logreg, "gradient_boosting": hgb}


def main() -> None:
    df = pd.read_parquet(FEATURES)
    df = df[df["label"] >= 0].copy()
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    train, test = time_split(df)
    x_train, y_train = train[NUMERIC + CATEGORICAL], train["label"].to_numpy()
    x_test, y_test = test[NUMERIC + CATEGORICAL], test["label"].to_numpy()

    results = {
        "data": {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_period": [str(train["created_at"].min().date()), str(train["created_at"].max().date())],
            "test_period": [str(test["created_at"].min().date()), str(test["created_at"].max().date())],
            "test_failure_rate": round(float(y_test.mean()), 4),
        },
        "models": {
            "base_rate": evaluate(y_test, np.full(len(y_test), y_train.mean())),
            "bizz_trust_rule": evaluate(y_test, test["trust_level"].map(TRUST_SCORE).astype(float).to_numpy()),
        },
    }

    scores = {}
    for name, model in models().items():
        model.fit(x_train, y_train)
        p = model.predict_proba(x_test)[:, 1]
        scores[name] = p
        results["models"][name] = evaluate(y_test, p) | {"brier": round(float(brier_score_loss(y_test, p)), 4)}

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    curves = {"bizz_trust_rule": test["trust_level"].map(TRUST_SCORE).astype(float).to_numpy(), **scores}
    for name, s in curves.items():
        prec, rec, _ = precision_recall_curve(y_test, s)
        ax.plot(rec, prec, label=f"{name} (PR-AUC {average_precision_score(y_test, s):.3f})")
    ax.axhline(y_test.mean(), ls="--", c="grey", lw=1, label="base rate")
    ax.set(
        xlabel="Recall (share of failed orders caught)",
        ylabel="Precision",
        title="Failed-order detection, test period",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=150)

    best = max(scores, key=lambda n: results["models"][n]["pr_auc"])
    keep = ["created_at", "tenant_id", "wilaya", "trust_level", "from_live", "items_value", "statut", "label"]
    out = test[keep].copy()
    out["risk_score"] = scores[best]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED_DIR / "test_scores.parquet", index=False)


if __name__ == "__main__":
    main()
