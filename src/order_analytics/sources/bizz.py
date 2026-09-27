"""Read-only, anonymized extraction of Bizz orders into data/raw/orders.parquet."""

import hashlib
import hmac
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from order_analytics.config import RAW_DIR, RAW_ORDERS

# Only analysis columns are selected. Names, phones, addresses, communes and notes never leave the DB.
QUERY = text("""
SELECT
  c.id                              AS order_id,
  c."tenantId"                      AS tenant_id,
  c."clientId"                      AS client_id,
  c."livreurId"                     AS livreur_id,
  c.statut,
  c.source,
  c."fraisLivraison"                AS delivery_fee,
  c."fraisPayePar"                  AS fee_paid_by,
  c.remise                          AS credit_discount,
  c.promo                           AS promo_discount,
  c."liveId" IS NOT NULL            AS from_live,
  c."sourceCommandeId" IS NOT NULL  AS is_exchange_order,
  c."yalidineTracking" IS NOT NULL  AS has_yalidine_tracking,
  c."createdAt"                     AS created_at,
  c."updatedAt"                     AS updated_at,
  cl.wilaya,
  cl."createdAt"                    AS client_created_at,
  COALESCE(a.n_items, 0)            AS n_items,
  COALESCE(a.items_value, 0)        AS items_value,
  a.items_cost
FROM "Commande" c
JOIN "Client" cl ON cl.id = c."clientId"
JOIN "Tenant" t ON t.id = c."tenantId"
LEFT JOIN (
  SELECT "commandeId",
         count(*)          AS n_items,
         sum("prixVente")  AS items_value,
         sum("prixAchat")  AS items_cost
  FROM "Article"
  GROUP BY "commandeId"
) a ON a."commandeId" = c.id
-- Real shops only: seeded demo shops (slug demo_*) and the test shop must never enter the analysis.
WHERE left(t.slug, 5) <> 'demo_' AND t.slug <> 'test'
ORDER BY c."createdAt"
""")


def pseudonymize(values: pd.Series, salt: bytes, prefix: str) -> pd.Series:
    def one(v):
        if pd.isna(v):
            return None
        digest = hmac.new(salt, str(int(v)).encode(), hashlib.sha256).hexdigest()[:12]
        return f"{prefix}_{digest}"

    return values.map(one)


def main() -> None:
    load_dotenv()
    url = os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+psycopg://", 1)
    salt = os.environ["PSEUDO_SALT"].encode()

    engine = create_engine(url)
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        df = pd.read_sql(QUERY, conn)

    df["tenant_id"] = pseudonymize(df["tenant_id"], salt, "t")
    df["client_id"] = pseudonymize(df["client_id"], salt, "c")
    df["livreur_id"] = pseudonymize(df["livreur_id"], salt, "l")
    df = df.drop(columns=["order_id"])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_ORDERS, index=False)
    print(f"{len(df)} orders -> {RAW_ORDERS}")
    print(df["statut"].value_counts().to_string())


if __name__ == "__main__":
    main()
