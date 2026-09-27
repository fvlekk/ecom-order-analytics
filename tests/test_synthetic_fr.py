from order_analytics.markets import fr
from order_analytics.synthetic_fr import generate


def test_calibration_matches_cited_french_figures():
    df = generate(n_customers=2500, seed=7)
    done = df[df["statut"] != "annulee"]
    assert 55 <= df["basket_eur"].mean() <= 72  # FEVAD 2025: panier moyen 62 €
    assert 0.15 <= (done["statut"] == "retournee").mean() <= 0.25  # clothing returns 18-23 %
    assert set(df["statut"]) <= {"livree", "retournee", "non_retiree", "annulee"}
    assert set(df["region"]) <= set(fr.REGIONS)
    assert df["synthetic"].all()


def test_relay_only_parcels_can_be_left_uncollected():
    df = generate(n_customers=2500, seed=7)
    assert set(df.loc[df["statut"] == "non_retiree", "delivery_mode"]) == {"Point relais"}
