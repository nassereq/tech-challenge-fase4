"""Construção do contexto e geração da resposta fundamentada (Claude)."""
from __future__ import annotations

import os
import re

from src.config import LLM_EFFORT, LLM_MAX_TOKENS, LLM_MODEL
from src.retriever import Evidence

SYSTEM_PROMPT = """Você é um analista de Voice of Customer de um e-commerce brasileiro.
Responda perguntas de gestores usando EXCLUSIVAMENTE as avaliações de clientes fornecidas em <evidencias>.

Regras:
1. Toda afirmação deve citar as avaliações que a sustentam no formato [R1], [R2]...
2. Não invente fatos, números, percentuais ou causas que não estejam nas evidências.
   As evidências são uma AMOSTRA recuperada por similaridade: fale em "padrões nas avaliações
   recuperadas", não em estatísticas da base inteira.
3. Agrupe os achados em temas (ex.: atraso, produto divergente, qualidade, atendimento) e indique
   quantas evidências sustentam cada tema.
4. Se as evidências não forem suficientes ou não tratarem da pergunta, ou se a pergunta estiver
   fora do escopo (avaliações de clientes do e-commerce), responda apenas:
   "SEM_EVIDENCIAS: <explicação curta do que falta>".
5. Responda em português, de forma objetiva, em markdown."""


def build_context(evidences: list[Evidence]) -> str:
    lines = []
    for e in evidences:
        m = e.metadata
        lines.append(
            f'<avaliacao ref="{e.ref}" nota="{m.get("review_score")}" '
            f'categoria="{m.get("category")}" compra="{m.get("purchase_month")}" '
            f'entrega="{m.get("delivery_status")}" uf="{m.get("customer_state")}">\n'
            f"{e.text}\n</avaliacao>")
    return "<evidencias>\n" + "\n".join(lines) + "\n</evidencias>"


def cited_refs(answer: str) -> list[str]:
    return sorted(set(re.findall(r"\[(R\d+)\]", answer)), key=lambda r: int(r[1:]))


def has_llm() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def generate(question: str, evidences: list[Evidence]) -> dict:
    """Chama o Claude com as evidências. Retorna texto, refs citadas e status."""
    import anthropic

    client = anthropic.Anthropic()
    user_msg = f"{build_context(evidences)}\n\n<pergunta>{question}</pergunta>"

    # fallbacks="default": se o modelo recusar por política de segurança, a própria API
    # repete a requisição num modelo de fallback (beta server-side-fallback).
    resp = client.beta.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": LLM_EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": user_msg}],
    )
    if resp.stop_reason == "refusal":
        return {"status": "refusal", "answer": "O modelo recusou a solicitação.", "cited": []}

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if text.startswith("SEM_EVIDENCIAS"):
        return {"status": "insufficient", "answer": text.split(":", 1)[-1].strip(), "cited": []}

    valid = {e.ref for e in evidences}
    cited = cited_refs(text)
    return {"status": "ok", "answer": text,
            "cited": [r for r in cited if r in valid],
            "invalid_citations": [r for r in cited if r not in valid],
            "usage": {"input_tokens": resp.usage.input_tokens,
                      "output_tokens": resp.usage.output_tokens}}
