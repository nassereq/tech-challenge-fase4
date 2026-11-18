"""Orquestração do pipeline RAG + CLI.

Uso:  python -m src.rag "Quais são os principais problemas relatados pelos clientes?"
"""
from __future__ import annotations

import sys

from src.config import MIN_EVIDENCES
from src.generator import generate, has_llm
from src.retriever import Retriever, transform_query

_retriever: Retriever | None = None

NO_EVIDENCE_MSG = ("Não encontrei avaliações suficientemente relevantes para responder a essa "
                   "pergunta com segurança. Ela pode estar fora do escopo da base (avaliações de "
                   "clientes do e-commerce) ou os filtros aplicados são restritivos demais.")


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def answer(question: str, filters: dict | None = None, infer_filters: bool = True,
           use_llm: bool = True) -> dict:
    plan = transform_query(question, filters, infer_filters)
    evidences, stats = get_retriever().retrieve(plan)
    result = {"question": question, "plan": plan, "stats": stats, "evidences": evidences}

    # Guarda 1: evidência insuficiente -> não chama o LLM (evita alucinação)
    if len(evidences) < MIN_EVIDENCES:
        return {**result, "status": "insufficient", "answer": NO_EVIDENCE_MSG, "cited": []}

    if not (use_llm and has_llm()):
        return {**result, "status": "retrieval_only", "cited": [],
                "answer": "Modo somente recuperação (sem ANTHROPIC_API_KEY): veja as evidências."}

    # Guarda 2: o próprio LLM pode declarar SEM_EVIDENCIAS
    return {**result, **generate(question, evidences)}


def _print(res: dict) -> None:
    print(f"\n=== {res['question']}")
    print(f"Filtros: {res['plan'].filters} | {'; '.join(res['plan'].notes)}")
    print(f"Recuperação: {res['stats']}")
    print(f"Status: {res['status']}\n\n{res['answer']}\n")
    print("--- Evidências")
    for e in res["evidences"]:
        mark = "*" if e.ref in res["cited"] else " "
        m = e.metadata
        print(f"{mark}[{e.ref}] sim={e.similarity:.3f} nota={m['review_score']} "
              f"{m['category']} | {e.text[:140]}")


if __name__ == "__main__":
    _print(answer(" ".join(sys.argv[1:]) or "Quais são os principais problemas relatados?"))
