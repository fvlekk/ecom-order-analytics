"""Power BI export: a star schema of CSV files with French column names and labels.

  fact_commandes_dz  real Bizz orders (anonymized; stays local, never committed)
  fact_commandes_fr  synthetic French B2C orders (labelled synthetic)
  fact_commandes_br  Olist orders (CC BY-NC-SA 4.0: derived data keeps that licence)
  dim_date           one calendar with Algerian and French flags
  dim_wilaya         58 wilayas with their region

CSV: UTF-8 with BOM (Excel-friendly), comma separator, dot decimals. In Power Query, import
with locale "English (United States)" so decimals are read correctly.
"""

import numpy as np
import pandas as pd

from order_analytics.analysis_dz import likely_swaps
from order_analytics.config import OLIST_ORDERS, PROCESSED_DIR, RAW_ORDERS, ROOT, SYNTHETIC_ORDERS, require
from order_analytics.features import build as build_dz
from order_analytics.markets import dz, fr
from order_analytics.olist_late import build as build_br
from order_analytics.synthetic_fr import FR_SYNTHETIC

OUT = ROOT / "data" / "powerbi"
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]
STATUT_DZ = {
    "LIVREE": "Livrée", "AVEC_ECHANGE": "Échangée", "ANNULEE": "Annulée", "RETOUR": "Retour",
    "RECUPERE": "Récupérée", "EN_ATTENTE": "En attente", "CONFIRMEE": "Confirmée", "EN_LIVRAISON": "En livraison",
}  # fmt: skip
STATUT_FR = {"livree": "Livrée", "retournee": "Retournée", "non_retiree": "Non retirée", "annulee": "Annulée"}
RESULTAT_FR = {"livree": "Gardée", "retournee": "Retournée", "non_retiree": "Non retirée", "annulee": "Annulée"}


def _date_key(local_ts: pd.Series) -> pd.Series:
    return local_ts.dt.tz_localize(None).dt.strftime("%Y%m%d").astype(int)


def _save(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"{name}: {len(df)} lignes")


def fact_dz() -> pd.DataFrame:
    # Real Bizz orders stay on the owner's machine; everyone else gets the synthetic ones (same schema).
    synthetic = not RAW_ORDERS.exists()
    src = (
        require(SYNTHETIC_ORDERS, "Run `uv run python -m order_analytics.synthetic` first.")
        if synthetic
        else RAW_ORDERS
    )
    print(f"fact_commandes_dz from {'synthetic' if synthetic else 'private'} orders")
    df = build_dz(pd.read_parquet(src))
    local = dz.to_local(df["created_at"])
    df["commande_n"] = df.groupby("client_id").cumcount() + 1
    cost_known = df["items_cost"].where(df["items_cost"] > 0)
    return pd.DataFrame({
        "date_key": _date_key(local),
        "client_key": df["client_id"],
        "wilaya_code": df["wilaya_code"],
        "statut": df["statut"].map(STATUT_DZ),
        "resultat": np.select([df["label"] == 0, df["label"] == 1], ["Livrée", "Échouée"], "En cours"),
        "source": df["source"].map({"BIZZ": "Saisie Bizz", "YALIDINE": "Import Yalidine"}).fillna(df["source"]),
        "transporteur": df["carrier"],
        "montant_articles_da": df["items_value"].round(0),
        "cout_connu_da": cost_known.round(0),
        "frais_livraison_da": df["delivery_fee"],
        "promo_da": df["promo_discount"],
        "nb_articles": df["n_items"],
        "depuis_live": np.where(df["from_live"] == 1, "Oui", "Non"),
        "heure": df["hour"],
        "commande_n_client": df["commande_n"],
        "taux_echec_route_60j": df["route_recent_fail_rate"].round(4),
        "echange_probable": np.where(likely_swaps(df), "Oui", "Non"),
        "synthetique": "Oui" if synthetic else "Non",
    })  # fmt: skip


def fact_fr() -> pd.DataFrame:
    df = pd.read_parquet(require(FR_SYNTHETIC, "Run `uv run python -m order_analytics.synthetic_fr` first."))
    local = fr.to_local(df["created_at"])
    return pd.DataFrame({
        "date_key": _date_key(local),
        "commande_id": df["order_id"],
        "client_key": df["customer_id"],
        "departement": df["department"],
        "region": df["region"],
        "statut": df["statut"].map(STATUT_FR),
        "resultat": df["statut"].map(RESULTAT_FR),
        "panier_eur": df["basket_eur"],
        "remise_eur": df["discount_eur"],
        "frais_port_eur": df["shipping_fee_eur"],
        "nb_articles": df["n_items"],
        "paiement": df["payment"],
        "mode_livraison": df["delivery_mode"],
        "delai_promis_j": df["promised_days"],
        "heure": local.dt.hour,
        "synthetique": "Oui",
    })  # fmt: skip


def fact_br() -> pd.DataFrame:
    df = build_br(pd.read_parquet(OLIST_ORDERS))
    scores_path = PROCESSED_DIR / "olist_scores_jun_aug_2018.parquet"
    if scores_path.exists():
        scores = pd.read_parquet(scores_path)[["order_id", "slack_rule_score", "model_score"]]
        df = df.merge(scores, on="order_id", how="left")
    else:
        df["slack_rule_score"] = df["model_score"] = np.nan
    delivered = df["order_delivered_customer_date"]
    return pd.DataFrame({
        "date_key": df["purchased_at"].dt.strftime("%Y%m%d").astype(int),
        "commande_id": df["order_id"],
        "etat_client": df["customer_state"],
        "etat_vendeur": df["seller_state"],
        "statut": df["order_status"],
        "en_retard": df["label"],
        "delai_promis_j": df["promised_days"].round(1),
        "delai_reel_j": ((delivered - df["purchased_at"]).dt.total_seconds() / 86400).round(1),
        "marge_promesse_j": df["promise_slack"].round(1),
        "distance_km": df["distance_km"].round(0),
        "montant_brl": df["items_value"].round(2),
        "fret_brl": df["freight_value"].round(2),
        "paiement": df["payment_type"],
        "categorie": df["main_category"],
        "note_avis": df["review_score"],
        "score_regle": df["slack_rule_score"],
        "score_modele": df["model_score"],
    })  # fmt: skip


def dim_date(keys: pd.Series) -> pd.DataFrame:
    days = pd.date_range(pd.to_datetime(str(keys.min())), pd.to_datetime(str(keys.max())), freq="D")
    dz_cal = dz.calendar_features(pd.Series(days).dt.tz_localize(dz.TZ))
    fr_cal = fr.calendar_features(pd.Series(days).dt.tz_localize(fr.TZ))
    iso = days.isocalendar()
    return pd.DataFrame({
        "date_key": days.strftime("%Y%m%d").astype(int),
        "date": days.date,
        "annee": days.year,
        "mois_num": days.month,
        "mois": [MOIS[m - 1] for m in days.month],
        "annee_mois": days.strftime("%Y-%m"),
        "semaine_iso": iso.week.to_numpy(),
        "jour_num": days.dayofweek + 1,
        "jour": [JOURS[d] for d in days.dayofweek],
        "weekend_dz": dz_cal["is_weekend_dz"].to_numpy(),
        "ramadan": dz_cal["in_ramadan"].to_numpy(),
        "avant_aid_14j": dz_cal["pre_eid"].to_numpy(),
        "weekend_fr": fr_cal["is_weekend_fr"].to_numpy(),
        "jour_ferie_fr": fr_cal["is_holiday"].to_numpy(),
        "soldes_fr": fr_cal["in_soldes"].to_numpy(),
        "black_friday": fr_cal["black_friday"].to_numpy(),
        "avant_noel": fr_cal["pre_christmas"].to_numpy(),
    })  # fmt: skip


def dim_wilaya() -> pd.DataFrame:
    codes = range(1, len(dz.WILAYAS) + 1)
    return pd.DataFrame({"wilaya_code": list(codes), "wilaya": dz.WILAYAS, "region": [dz.region(c) for c in codes]})


def main() -> None:
    facts = {"fact_commandes_dz": fact_dz(), "fact_commandes_fr": fact_fr()}
    if OLIST_ORDERS.exists():
        facts["fact_commandes_br"] = fact_br()
    else:
        print("fact_commandes_br skipped: run `uv run python -m order_analytics.sources.olist` first.")
    for name, df in facts.items():
        _save(df, name)
    _save(dim_date(pd.concat([f["date_key"] for f in facts.values()])), "dim_date")
    _save(dim_wilaya(), "dim_wilaya")
    (OUT / "LICENCE_olist.txt").write_text(
        "fact_commandes_br.csv is derived from the Olist Brazilian E-Commerce Public Dataset\n"
        "(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), licence CC BY-NC-SA 4.0.\n"
        "Changes: aggregated to one row per order, derived columns added. Non-commercial use only.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
