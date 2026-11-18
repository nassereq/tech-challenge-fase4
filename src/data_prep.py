"""Construção da base de conhecimento: seleção, limpeza e organização dos documentos.

Cada documento = 1 avaliação com comentário textual, enriquecida com metadados
(nota, datas, categoria, UF, atraso na entrega) usados como filtros na recuperação.

Uso:  python -m src.data_prep
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED, DOCS_PARQUET

_WS = re.compile(r"\s+")
_HAS_LETTER = re.compile(r"[a-zA-ZÀ-ÿ]")


def clean_text(value) -> str:
    """Normaliza o texto sem destruir o conteúdo semântico.

    - NaN/None -> "" (tratamento de ausentes)
    - normalização Unicode NFC (acentos consistentes)
    - quebras de linha / espaços múltiplos -> espaço simples
    Mantemos caixa, acentos e pontuação: o modelo de embeddings é
    multilíngue/subword e se beneficia do texto original (stemming ou remoção
    de stopwords prejudicariam a semântica, ex.: "não chegou").
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return _WS.sub(" ", text).strip()


def load_raw() -> dict[str, pd.DataFrame]:
    files = {
        "reviews": "olist_order_reviews_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "translation": "product_category_name_translation.csv",
        "customers": "olist_customers_dataset.csv",
    }
    missing = [f for f in files.values() if not (DATA_RAW / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"CSVs ausentes em {DATA_RAW}: {missing}. Rode scripts/download_data.py"
        )
    return {k: pd.read_csv(DATA_RAW / f) for k, f in files.items()}


def order_category(items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Categoria predominante por pedido (pedidos podem ter vários itens)."""
    df = items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    df["product_category_name"] = df["product_category_name"].fillna("desconhecida")
    cat = (
        df.groupby(["order_id", "product_category_name"]).size()
        .reset_index(name="n")
        .sort_values(["order_id", "n"], ascending=[True, False])
        .drop_duplicates("order_id")
    )
    agg = items.groupby("order_id").agg(n_items=("order_item_id", "count"),
                                        order_value=("price", "sum"),
                                        freight_value=("freight_value", "sum"))
    return cat[["order_id", "product_category_name"]].merge(agg, on="order_id", how="left")


def build_documents() -> pd.DataFrame:
    raw = load_raw()
    rv = raw["reviews"].copy()
    n_total = len(rv)

    # 1) Limpeza e tratamento de ausentes ------------------------------------
    rv["title"] = rv["review_comment_title"].map(clean_text)
    rv["message"] = rv["review_comment_message"].map(clean_text)

    # 2) Seleção: só avaliações com comentário textual útil -------------------
    # Avaliações sem texto (≈58% da base) não carregam a "voz do cliente".
    rv = rv[rv["message"].str.len() >= 3]
    rv = rv[rv["message"].str.contains(_HAS_LETTER)]  # descarta "...", "!!!", "10"

    # 3) Deduplicação: o mesmo review_id pode estar ligado a vários pedidos ---
    rv = rv.sort_values("review_creation_date").drop_duplicates("review_id", keep="first")

    # 4) Enriquecimento com metadados ----------------------------------------
    orders = raw["orders"]
    for col in ["order_purchase_timestamp", "order_delivered_customer_date",
                "order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    orders = orders.merge(raw["customers"][["customer_id", "customer_state"]],
                          on="customer_id", how="left")

    cats = order_category(raw["items"], raw["products"]).merge(
        raw["translation"], on="product_category_name", how="left")

    df = (rv.merge(orders, on="order_id", how="left")
            .merge(cats, on="order_id", how="left"))

    delivered = df["order_delivered_customer_date"]
    estimated = df["order_estimated_delivery_date"]
    df["delivery_delay_days"] = (delivered - estimated).dt.days
    # Situação de entrega: atrasado / no prazo / não entregue (ausente)
    df["delivery_status"] = "nao_entregue"
    df.loc[delivered.notna() & (df["delivery_delay_days"] > 0), "delivery_status"] = "atrasado"
    df.loc[delivered.notna() & (df["delivery_delay_days"] <= 0), "delivery_status"] = "no_prazo"

    df["sentiment_bucket"] = pd.cut(df["review_score"], bins=[0, 2, 3, 5],
                                    labels=["negativo", "neutro", "positivo"]).astype(str)
    purchase = df["order_purchase_timestamp"]
    df["purchase_year"] = purchase.dt.year.fillna(0).astype(int)
    df["purchase_month"] = purchase.dt.strftime("%Y-%m").fillna("desconhecido")
    df["category"] = df["product_category_name"].fillna("desconhecida")
    df["category_en"] = df["product_category_name_english"].fillna("unknown")
    df["customer_state"] = df["customer_state"].fillna("NA")
    df["order_status"] = df["order_status"].fillna("desconhecido")

    # 5) Texto do documento (o que vai para o embedding) ----------------------
    df["text"] = df.apply(
        lambda r: f"{r['title']}. {r['message']}" if r["title"] else r["message"], axis=1)

    cols = ["review_id", "order_id", "text", "title", "message", "review_score",
            "sentiment_bucket", "review_creation_date", "purchase_year", "purchase_month",
            "category", "category_en", "customer_state", "order_status",
            "delivery_status", "delivery_delay_days", "n_items", "order_value", "freight_value"]
    out = df[cols].reset_index(drop=True)

    print(f"Avaliações totais: {n_total:,} | documentos com texto útil: {len(out):,} "
          f"({len(out) / n_total:.1%})")
    print(out["review_score"].value_counts().sort_index().to_string())
    return out


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    docs = build_documents()
    docs.to_parquet(DOCS_PARQUET, index=False)
    print(f"Salvo em {DOCS_PARQUET}")


if __name__ == "__main__":
    main()
