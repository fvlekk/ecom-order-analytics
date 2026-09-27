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

OLIST_HINT = "Run `uv run python -m order_analytics.sources.olist` first."
PRIVATE_HINT = "This is private Bizz data (sources.bizz with a DATABASE_URL); it is not part of the public repo."


def require(path: Path, how: str) -> Path:
    """Stop with a clear instruction instead of a traceback when an input file is missing."""
    if not path.exists():
        raise SystemExit(f"Missing {path.relative_to(ROOT)}. {how}")
    return path
