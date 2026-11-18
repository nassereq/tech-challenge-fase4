"""Construção do contexto e geração da resposta fundamentada (Gemini ou Claude)."""
from __future__ import annotations

import os
import re

from src.config import GEMINI_MODELS, LLM_EFFORT, LLM_MAX_TOKENS, LLM_MODEL
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
    # aceita [R1] e também [R1, R2, R5]
    refs = {r for grp in re.findall(r"\[(R\d+(?:\s*,\s*R\d+)*)\]", answer)
            for r in re.findall(r"R\d+", grp)}
    return sorted(refs, key=lambda r: int(r[1:]))


def provider() -> str | None:
    """Gemini (cota gratuita) tem prioridade; Claude se houver chave da Anthropic."""
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return "claude"
    return None


def has_llm() -> bool:
    return provider() is not None


def _call_gemini(user_msg: str) -> tuple[str, dict]:
    from google import genai
    from google.genai import types

    client = genai.Client()  # lê GEMINI_API_KEY do ambiente
    # Sem retentativas longas: se um modelo estiver sobrecarregado (503), passa ao próximo.
    http = types.HttpOptions(timeout=90_000, retry_options=types.HttpRetryOptions(attempts=1))
    last_err = None
    for model in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(
                model=model, contents=user_msg,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
                                                   temperature=0.2, http_options=http))
            return (resp.text or "").strip(), {"model": model}
        except Exception as e:  # 503/404/timeout -> tenta o próximo modelo
            last_err = e
    raise RuntimeError(f"Nenhum modelo Gemini disponível no momento: {last_err}")


def generate(question: str, evidences: list[Evidence]) -> dict:
    """Chama o LLM com as evidências. Retorna texto, refs citadas e status."""
    user_msg = f"{build_context(evidences)}\n\n<pergunta>{question}</pergunta>"
    if provider() == "gemini":
        text, usage = _call_gemini(user_msg)
        return _finish(text, evidences, usage)

    import anthropic

    client = anthropic.Anthropic()

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
    return _finish(text, evidences, {"input_tokens": resp.usage.input_tokens,
                                     "output_tokens": resp.usage.output_tokens})


def _finish(text: str, evidences: list[Evidence], usage: dict) -> dict:
    if text.startswith("SEM_EVIDENCIAS"):
        return {"status": "insufficient", "answer": text.split(":", 1)[-1].strip(), "cited": []}

    valid = {e.ref for e in evidences}
    cited = cited_refs(text)
    return {"status": "ok", "answer": text,
            "cited": [r for r in cited if r in valid],
            "invalid_citations": [r for r in cited if r not in valid],
            "usage": usage}
