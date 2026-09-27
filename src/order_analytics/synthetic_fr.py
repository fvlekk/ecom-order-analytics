"""Synthetic B2C fashion orders for the French market. NOT real data.

No public French order-level B2C dataset exists, so this generator demonstrates the FR market
module and feeds the Power BI demo. Calibration, and what is only an assumption:
- Average basket ~ €62-70: FEVAD, bilan e-commerce 2025 (panier moyen 62 €).
- Clothing return rate ~ 18-23%: FashionNetwork / Statista (2022-2023). Target here: ~20%.
- Regional weights: INSEE population by region (approximate, 2021).
- ASSUMPTION (no public source found): payment mix, delivery-mode mix, share of relay parcels never
  collected (~3%), cancellation before shipping (~1.5%), sales/Black Friday volume multipliers,
  and every effect size on returns. Never draw business conclusions from this data.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from order_analytics.config import RAW_DIR, SEED
from order_analytics.markets import fr

FR_SYNTHETIC = RAW_DIR / "orders_fr_synthetic.parquet"

# INSEE 2021, millions of inhabitants (approximate).
POPULATION = {
    "Île-de-France": 12.3, "Auvergne-Rhône-Alpes": 8.1, "Nouvelle-Aquitaine": 6.0, "Occitanie": 6.0,
    "Hauts-de-France": 6.0, "Grand Est": 5.6, "Provence-Alpes-Côte d'Azur": 5.1, "Pays de la Loire": 3.8,
    "Bretagne": 3.4, "Normandie": 3.3, "Bourgogne-Franche-Comté": 2.8, "Centre-Val de Loire": 2.6, "Corse": 0.35,
}  # fmt: skip
PAYMENTS = (["Carte bancaire", "PayPal", "Apple Pay", "Paiement en 3x"], [0.76, 0.16, 0.06, 0.02])  # ASSUMPTION
DELIVERY = {  # mode: (share, fee under the free-shipping threshold, promised days) — ASSUMPTION
    "Point relais": (0.45, 3.99, 4),
    "Domicile": (0.45, 5.99, 3),
    "Express": (0.10, 9.99, 1),
}
FREE_SHIPPING_FROM = 60.0
_HOUR_WEIGHTS = np.array(
    [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 6, 5, 5, 6, 7, 8, 9, 9, 8, 5, 3]
)  # ASSUMPTION: evening peak
P_HOUR = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()


def _volume_multiplier(d: date, sales: list[tuple[date, date]], bf: dict[int, date]) -> float:
    m = 1.0
    if any(s <= d <= e for s, e in sales):
        m *= 1.6
    if 0 <= (d - bf[d.year]).days <= 3:
        m *= 3.0
    if d.month == 12 and d.day <= 20:
        m *= 1.5
    if d.month == 8:
        m *= 0.8
    return m


def generate(
    n_customers: int = 9000, start: str = "2024-01-01", end: str = "2025-12-31", seed: int = SEED
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    years = range(d0.year, d1.year + 1)
    sales = [p for y in years for p in fr.sales_periods(y)]
    bf = {y: fr.black_friday(y) for y in years}

    regions = list(POPULATION)
    p_region = np.array([POPULATION[r] for r in regions]) / sum(POPULATION.values())
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    p_day = np.array([_volume_multiplier(d, sales, bf) for d in days])
    p_day /= p_day.sum()
    modes = list(DELIVERY)
    p_mode = [DELIVERY[m][0] for m in modes]

    rows = []
    for c in range(n_customers):
        reg = str(rng.choice(regions, p=p_region))
        dept = str(rng.choice(fr.REGIONS[reg]))
        habit = rng.normal(0, 0.8)  # hidden tendency to return
        n_orders = int(min(rng.geometric(0.55), 12))
        for k, day in enumerate(sorted(rng.choice(len(days), size=n_orders, p=p_day))):
            d = days[day]
            in_sales = any(s <= d <= e for s, e in sales)
            is_bf = 0 <= (d - bf[d.year]).days <= 3
            n_items = int(rng.choice([1, 2, 3, 4], p=[0.55, 0.28, 0.12, 0.05]))
            unit = rng.lognormal(np.log(37), 0.45, n_items)
            gross = float(unit.sum())
            discount = gross * (0.30 if in_sales else 0.25 if is_bf else 0.0)
            basket = round(gross - discount, 2)
            mode = str(rng.choice(modes, p=p_mode))
            fee = 0.0 if basket >= FREE_SHIPPING_FROM else DELIVERY[mode][1]
            hour = int(rng.choice(24, p=P_HOUR))
            created_local = pd.Timestamp(d) + pd.Timedelta(hours=hour, minutes=int(rng.integers(60)))

            logit = (
                -1.97 + habit + 0.6 * (n_items >= 2) + 0.2 * in_sales + 0.3 * is_bf
                + 0.3 * (k == 0) - 0.3 * (k >= 3) - 0.1 * (mode == "Express") + 0.004 * (gross / n_items - 37)
            )  # fmt: skip
            r = rng.random()
            if r < 0.015:
                statut = "annulee"
            elif mode == "Point relais" and r < 0.015 + 0.03:
                statut = "non_retiree"
            elif rng.random() < 1 / (1 + np.exp(-logit)):
                statut = "retournee"
            else:
                statut = "livree"

            rows.append(
                {
                    "order_id": f"FR{len(rows) + 1:06d}",
                    "customer_id": f"CFR{c:05d}",
                    "created_at": created_local.tz_localize(fr.TZ, ambiguous=False, nonexistent="shift_forward")
                    .tz_convert("UTC")
                    .tz_localize(None),
                    "department": dept,
                    "region": reg,
                    "n_items": n_items,
                    "basket_eur": basket,
                    "discount_eur": round(discount, 2),
                    "payment": str(rng.choice(PAYMENTS[0], p=PAYMENTS[1])),
                    "delivery_mode": mode,
                    "shipping_fee_eur": fee,
                    "promised_days": DELIVERY[mode][2],
                    "statut": statut,
                    "synthetic": True,
                }
            )
    return pd.DataFrame(rows).sort_values("created_at", ignore_index=True)


def main() -> None:
    df = generate()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FR_SYNTHETIC, index=False)
    done = df[df["statut"] != "annulee"]
    print(f"{len(df)} synthetic FR orders -> {FR_SYNTHETIC}")
    print(f"mean basket €{df['basket_eur'].mean():.2f} | return rate {(done['statut'] == 'retournee').mean():.1%}")
    print(df["statut"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
