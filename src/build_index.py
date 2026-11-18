"""Gera embeddings dos documentos e indexa no ChromaDB (persistente em disco).

Uso:  python -m src.build_index [--limit N]
"""
from __future__ import annotations

import argparse

import chromadb
import pandas as pd
from tqdm import tqdm

from src.config import (CHROMA_DIR, COLLECTION_NAME, DOCS_PARQUET, PASSAGE_PREFIX)
from src.embeddings import get_embedder

META_COLS = ["order_id", "review_score", "sentiment_bucket", "review_creation_date",
             "purchase_year", "purchase_month", "category", "category_en",
             "customer_state", "order_status", "delivery_status"]


def to_metadata(row: pd.Series) -> dict:
    meta = {}
    for c in META_COLS:
        v = row[c]
        meta[c] = int(v) if c in ("review_score", "purchase_year") else str(v)
    delay = row["delivery_delay_days"]
    meta["delivery_delay_days"] = int(delay) if pd.notna(delay) else -999
    return meta


def build(limit: int | None = None, batch_size: int = 256) -> None:
    docs = pd.read_parquet(DOCS_PARQUET)
    if limit:
        docs = docs.sample(limit, random_state=42)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    model = get_embedder()
    for start in tqdm(range(0, len(docs), batch_size), desc="Indexando"):
        chunk = docs.iloc[start:start + batch_size]
        emb = model.encode([PASSAGE_PREFIX + t for t in chunk["text"]],
                           normalize_embeddings=True, batch_size=64)
        col.add(ids=chunk["review_id"].tolist(),
                documents=chunk["text"].tolist(),
                embeddings=emb.tolist(),
                metadatas=[to_metadata(r) for _, r in chunk.iterrows()])
    print(f"{col.count():,} documentos indexados em {CHROMA_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="indexar só uma amostra (teste)")
    build(ap.parse_args().limit)
