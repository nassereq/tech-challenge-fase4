"""Etapas de recuperação do pipeline RAG.

pergunta -> transform_query (normalização + filtros de metadados inferidos)
         -> embedding (prefixo "query: ")
         -> busca vetorial no ChromaDB (top CANDIDATES_K, com filtros)
         -> corte por similaridade mínima (evidência fraca é descartada)
         -> deduplicação de textos idênticos
         -> MMR (relevância x diversidade) -> FINAL_K evidências
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import chromadb
import numpy as np

from src.config import (CANDIDATES_K, CHROMA_DIR, COLLECTION_NAME, FINAL_K, MIN_SIMILARITY,
                        MMR_LAMBDA, QUERY_PREFIX)
from src.embeddings import get_embedder

# Regras simples (e auditáveis) para inferir filtros a partir da pergunta.
_NEG = re.compile(r"insatisf|nota[s]? baixa|negativ|reclama|detrat|piores?\b", re.I)
_POS = re.compile(r"\belogi|\bpositiv|\bsatisfeit|\bvaloriz|nota[s]? alta|\bmelhores?\b|\bgost", re.I)
_LATE = re.compile(r"atras", re.I)


@dataclass
class QueryPlan:
    original: str
    search_text: str
    filters: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    ref: str            # rótulo usado na citação, ex.: "R3"
    review_id: str
    text: str
    similarity: float
    metadata: dict


def transform_query(question: str, user_filters: dict | None = None,
                    infer_filters: bool = True) -> QueryPlan:
    q = re.sub(r"\s+", " ", question).strip()
    plan = QueryPlan(original=question, search_text=q)
    filters = dict(user_filters or {})

    if infer_filters and "review_score" not in filters and "sentiment_bucket" not in filters:
        if _NEG.search(q):
            filters["sentiment_bucket"] = "negativo"
            plan.notes.append("Pergunta sobre insatisfação → filtro nota ≤ 2")
        elif _POS.search(q):
            filters["sentiment_bucket"] = "positivo"
            plan.notes.append("Pergunta sobre elogios → filtro nota ≥ 4")
    if infer_filters and "delivery_status" not in filters and _LATE.search(q):
        plan.notes.append("Tema 'atraso' detectado (sem filtro rígido: a busca semântica cobre)")

    plan.filters = filters
    return plan


def _where(filters: dict) -> dict | None:
    clauses = []
    for k, v in filters.items():
        if v in (None, "", []):
            continue
        if k == "min_score":
            clauses.append({"review_score": {"$gte": int(v)}})
        elif k == "max_score":
            clauses.append({"review_score": {"$lte": int(v)}})
        elif isinstance(v, (list, tuple)):
            clauses.append({k: {"$in": list(v)}})
        else:
            clauses.append({k: v})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance: evita 15 evidências dizendo a mesma coisa."""
    rel = doc_vecs @ query_vec
    selected: list[int] = []
    remaining = list(range(len(doc_vecs)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: rel[i])
        else:
            sim_sel = doc_vecs[remaining] @ doc_vecs[selected].T
            scores = lam * rel[remaining] - (1 - lam) * sim_sel.max(axis=1)
            best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return selected


class Retriever:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = client.get_collection(COLLECTION_NAME)
        self.model = get_embedder()

    def retrieve(self, plan: QueryPlan, k: int = FINAL_K,
                 min_similarity: float = MIN_SIMILARITY) -> tuple[list[Evidence], dict]:
        qvec = self.model.encode(QUERY_PREFIX + plan.search_text, normalize_embeddings=True)
        res = self.col.query(query_embeddings=[qvec.tolist()], n_results=CANDIDATES_K,
                             where=_where(plan.filters),
                             include=["documents", "metadatas", "distances", "embeddings"])

        ids, docs = res["ids"][0], res["documents"][0]
        metas, dists = res["metadatas"][0], res["distances"][0]
        embs = np.asarray(res["embeddings"][0]) if len(ids) else np.empty((0, len(qvec)))
        sims = [1 - d for d in dists]

        # corte por similaridade + deduplicação de textos idênticos
        keep, seen = [], set()
        for i, (doc, s) in enumerate(zip(docs, sims)):
            norm = re.sub(r"\W+", " ", doc.lower()).strip()
            if s >= min_similarity and norm not in seen:
                seen.add(norm)
                keep.append(i)

        order = [keep[j] for j in _mmr(qvec, embs[keep], k, MMR_LAMBDA)] if keep else []
        evidences = [Evidence(ref=f"R{n + 1}", review_id=ids[i], text=docs[i],
                              similarity=round(sims[i], 4), metadata=metas[i])
                     for n, i in enumerate(order)]
        stats = {"candidates": len(ids), "above_threshold": len(keep),
                 "selected": len(evidences),
                 "max_similarity": round(max(sims), 4) if sims else None}
        return evidences, stats
