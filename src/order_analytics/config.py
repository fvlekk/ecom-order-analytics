from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

RAW_ORDERS = RAW_DIR / "orders.parquet"
SYNTHETIC_ORDERS = RAW_DIR / "orders_synthetic.parquet"
OLIST_ORDERS = RAW_DIR / "olist_orders.parquet"
OLIST_REPORTS = ROOT / "reports" / "olist"
FEATURES = PROCESSED_DIR / "features.parquet"

# Mirrors CANCEL_STATES in bizz/server/src/lib/stock.js: stock goes back to the shop.
FAILED = {"RETOUR", "ANNULEE", "RECUPERE"}
SUCCEEDED = {"LIVREE", "AVEC_ECHANGE"}
IN_PROGRESS = {"EN_ATTENTE", "CONFIRMEE", "EN_LIVRAISON"}

SEED = 42
