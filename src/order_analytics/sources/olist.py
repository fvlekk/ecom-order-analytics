"""Olist Brazilian E-Commerce Public Dataset: ~100k real B2C marketplace orders, 2016-2018.

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce (download needs a Kaggle account).
Put the zip at data/raw/public/olist_brazilian_ecommerce.zip; this module builds one row per order.
"""

import hashlib
import zipfile

import numpy as np
import pandas as pd

from order_analytics.config import OLIST_ORDERS, RAW_DIR

ZIP_PATH = RAW_DIR / "public" / "olist_brazilian_ecommerce.zip"
SHA256 = "967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784"

DATES = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def _read(z: zipfile.ZipFile, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(
        z.open(f"olist_{name}_dataset.csv" if name != "translation" else "product_category_name_translation.csv"), **kw
    )


def _zip_centroids(geo: pd.DataFrame) -> pd.DataFrame:
    # Raw geolocation has outliers outside Brazil; keep points inside its bounding box.
    geo = geo[geo["geolocation_lat"].between(-34, 6) & geo["geolocation_lng"].between(-74, -34)]
    return geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()


def _haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build(z: zipfile.ZipFile) -> pd.DataFrame:
    orders = _read(z, "orders", parse_dates=DATES)
    customers = _read(z, "customers")
    items = _read(z, "order_items")
    payments = _read(z, "order_payments")
    reviews = _read(z, "order_reviews")
    products = _read(z, "products").merge(_read(z, "translation"), on="product_category_name", how="left")
    sellers = _read(z, "sellers")
    centroids = _zip_centroids(_read(z, "geolocation"))

    items = items.merge(products, on="product_id", how="left").merge(sellers, on="seller_id", how="left")
    per_order = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        items_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        n_sellers=("seller_id", "nunique"),
        weight_g=("product_weight_g", "sum"),
    )
    # The most expensive item stands for the order's category and seller.
    main = items.sort_values("price", ascending=False).drop_duplicates("order_id").set_index("order_id")
    per_order["main_category"] = main["product_category_name_english"]
    per_order["seller_id"] = main["seller_id"]
    per_order["seller_state"] = main["seller_state"]
    per_order["seller_zip"] = main["seller_zip_code_prefix"]

    pay = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        n_payments=("payment_sequential", "size"),
    )
    pay["payment_type"] = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")
        .set_index("order_id")["payment_type"]
    )

    # Reviews are written after delivery: analysis only, never a model feature.
    review = reviews.sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")
    review = review.set_index("order_id")[["review_score"]]

    df = (
        orders.merge(customers, on="customer_id", how="left")
        .join(per_order, on="order_id")
        .join(pay, on="order_id")
        .join(review, on="order_id")
    )
    cust = centroids.reindex(df["customer_zip_code_prefix"]).to_numpy()
    sell = centroids.reindex(df["seller_zip"]).to_numpy()
    df["distance_km"] = _haversine_km(cust[:, 0], cust[:, 1], sell[:, 0], sell[:, 1])
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype("Int8").where(df["seller_state"].notna())

    return df.drop(columns=["customer_id", "seller_zip"]).rename(
        columns={"customer_zip_code_prefix": "customer_zip", "order_purchase_timestamp": "purchased_at"}
    )


def main() -> None:
    data = ZIP_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError(f"Checksum mismatch for {ZIP_PATH}: not the pinned Olist release")
    with zipfile.ZipFile(ZIP_PATH) as z:
        df = build(z)
    df.to_parquet(OLIST_ORDERS, index=False)
    print(f"{len(df)} orders -> {OLIST_ORDERS}")


if __name__ == "__main__":
    main()
