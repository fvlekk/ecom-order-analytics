"""Synthetic orders with the exact schema of sources/bizz.py.

Effect sizes are invented. Use this to develop the pipeline and in the public repo;
never draw business conclusions from it.
"""

import numpy as np
import pandas as pd

from order_analytics.config import RAW_DIR, SEED, SYNTHETIC_ORDERS

# (wilaya, population weight, extra log-odds of failure)
WILAYAS = [
    ("Alger", 12, -0.30),
    ("Oran", 8, -0.20),
    ("Constantine", 6, -0.15),
    ("Sétif", 6, 0.00),
    ("Blida", 5, -0.10),
    ("Batna", 4, 0.05),
    ("Béjaïa", 4, 0.00),
    ("Tizi Ouzou", 4, -0.05),
    ("Annaba", 4, -0.10),
    ("Djelfa", 4, 0.25),
    ("Tlemcen", 3, 0.05),
    ("Biskra", 3, 0.20),
    ("Chlef", 3, 0.10),
    ("Boumerdès", 3, -0.05),
    ("Médéa", 3, 0.10),
    ("Mostaganem", 2, 0.10),
    ("Ouargla", 2, 0.40),
    ("Ghardaïa", 1, 0.35),
    ("Béchar", 1, 0.50),
    ("Tamanrasset", 1, 0.70),
]


def generate(n_clients: int = 3000, n_tenants: int = 3, days: int = 540, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01", tz="UTC")

    names, weights, risk = zip(*WILAYAS, strict=True)
    p_wilaya = np.array(weights) / sum(weights)
    wilaya_risk = dict(zip(names, risk, strict=True))
    livreurs = {t: [f"l_syn{t}{i}" for i in range(4)] for t in range(n_tenants)}

    rows = []
    for c in range(n_clients):
        tenant = int(rng.integers(n_tenants))
        wilaya = str(rng.choice(names, p=p_wilaya))
        reliability = rng.normal(0, 1.1)  # hidden per-client tendency to refuse
        n_orders = int(min(rng.geometric(0.45), 25))
        first = start + pd.Timedelta(days=float(rng.uniform(0, days * 0.9)))
        t = first
        for _ in range(n_orders):
            if t > start + pd.Timedelta(days=days):
                break
            from_live = rng.random() < 0.45
            own_livreur = rng.random() < 0.35
            n_items = int(rng.integers(1, 5))
            unit = rng.normal(3200, 900, n_items).clip(800)
            value = float(unit.sum())
            fee = 0.0 if own_livreur else float(rng.choice([400, 500, 600, 800, 1000]))
            fee_paid_by = "BOUTIQUE" if rng.random() < 0.2 else "CLIENT"
            promo = float(rng.choice([0, 0, 0, 200, 500]))
            logit = (
                -1.25
                + 0.9 * reliability
                + wilaya_risk[wilaya]
                + 0.45 * from_live
                + 0.00012 * (value - 6000)
                + (0.25 if fee_paid_by == "CLIENT" and fee >= 800 else 0.0)
                - 0.15 * (promo > 0)
            )
            failed = rng.random() < 1 / (1 + np.exp(-logit))
            age_days = (start + pd.Timedelta(days=days) - t).days
            if age_days < 10 and rng.random() < 0.7:
                statut = str(rng.choice(["EN_ATTENTE", "CONFIRMEE", "EN_LIVRAISON"]))
            elif failed:
                statut = str(rng.choice(["RETOUR", "ANNULEE", "RECUPERE"], p=[0.55, 0.40, 0.05]))
            else:
                statut = "AVEC_ECHANGE" if rng.random() < 0.08 else "LIVREE"
            rows.append(
                {
                    "tenant_id": f"t_syn{tenant}",
                    "client_id": f"c_syn{c:05d}",
                    "livreur_id": str(rng.choice(livreurs[tenant])) if own_livreur else None,
                    "statut": statut,
                    "source": "BIZZ",
                    "delivery_fee": fee,
                    "fee_paid_by": fee_paid_by,
                    "credit_discount": 0.0,
                    "promo_discount": promo,
                    "from_live": from_live,
                    "is_exchange_order": rng.random() < 0.03,
                    "has_yalidine_tracking": not own_livreur,
                    "created_at": t,
                    "updated_at": t + pd.Timedelta(days=float(rng.uniform(1, 6))),
                    "wilaya": wilaya,
                    "client_created_at": first,
                    "n_items": n_items,
                    "items_value": value,
                    "items_cost": float(value * rng.uniform(0.45, 0.65)),
                }
            )
            t = t + pd.Timedelta(days=float(rng.exponential(35)))

    return pd.DataFrame(rows).sort_values("created_at", ignore_index=True)


def main() -> None:
    df = generate()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SYNTHETIC_ORDERS, index=False)
    print(f"{len(df)} synthetic orders -> {SYNTHETIC_ORDERS}")
    print(df["statut"].value_counts().to_string())


if __name__ == "__main__":
    main()
