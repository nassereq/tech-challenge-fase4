"""Roda perguntas de referência (dentro e fora do escopo) e salva os resultados.

Serve para (a) calibrar MIN_SIMILARITY comparando similaridades de perguntas
dentro vs. fora do escopo e (b) gerar os exemplos de perguntas/respostas do README.

Uso:
  python scripts/evaluate.py              # só recuperação (sem custo de LLM)
  python scripts/evaluate.py --llm        # com geração (usa a API da Anthropic)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag import answer  # noqa: E402

IN_SCOPE = [
    "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?",
    "Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?",
]
OUT_OF_SCOPE = [
    "Qual é a capital da Austrália?",
    "Como funciona a fotossíntese?",
    "Qual a previsão do tempo para amanhã em São Paulo?",
    "O que os clientes acham do atendimento por chatbot com inteligência artificial?",
]


def main(use_llm: bool) -> None:
    lines = ["# Resultados de avaliação\n",
             "| escopo | pergunta | max_sim | acima do limiar | status |", "|---|---|---|---|---|"]
    details = []
    for scope, qs in (("dentro", IN_SCOPE), ("fora", OUT_OF_SCOPE)):
        for q in qs:
            r = answer(q, use_llm=use_llm)
            s = r["stats"]
            print(f"[{scope}] max={s['max_similarity']} acima={s['above_threshold']} "
                  f"status={r['status']} | {q}")
            lines.append(f"| {scope} | {q} | {s['max_similarity']} | {s['above_threshold']} "
                         f"| {r['status']} |")
            ev = "\n".join(f"- **[{e.ref}]** (nota {e.metadata['review_score']}, "
                           f"{e.metadata['category']}, sim {e.similarity:.3f}) {e.text}"
                           for e in r["evidences"])
            details.append(f"## {q}\n\n**Filtros:** `{r['plan'].filters}`  \n"
                           f"**Status:** {r['status']}\n\n{r['answer']}\n\n"
                           f"<details><summary>Evidências</summary>\n\n{ev}\n\n</details>\n")
    out = ROOT / "reports" / ("exemplos_com_llm.md" if use_llm else "avaliacao_recuperacao.md")
    out.write_text("\n".join(lines) + "\n\n" + "\n".join(details), encoding="utf-8")
    print(f"Salvo em {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    main(ap.parse_args().llm)
