"""Interface Streamlit — Voice of Customer Intelligence (RAG sobre avaliações Olist).

Uso:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.config import DOCS_PARQUET
from src.generator import has_llm
from src.rag import answer

st.set_page_config(page_title="VoC Intelligence — Olist", layout="wide")
st.title("Voice of Customer Intelligence")
st.caption("RAG sobre avaliações de clientes do Brazilian E-Commerce Public Dataset by Olist")

EXAMPLES = [
    "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam que existem problemas recorrentes com o prazo de entrega?",
    "Qual é a capital da Austrália?",  # fora do escopo
]


@st.cache_data
def categories() -> list[str]:
    try:
        return sorted(pd.read_parquet(DOCS_PARQUET, columns=["category"])["category"].unique())
    except Exception:
        return []


with st.sidebar:
    st.header("Filtros (metadados)")
    infer = st.checkbox("Inferir filtros da pergunta", value=True,
                        help="Ex.: 'insatisfeitos' → nota ≤ 2; 'elogios' → nota ≥ 4")
    score = st.slider("Nota da avaliação", 1, 5, (1, 5))
    cats = st.multiselect("Categoria do produto", categories())
    delivery = st.multiselect("Situação da entrega", ["atrasado", "no_prazo", "nao_entregue"])
    years = st.multiselect("Ano da compra", [2016, 2017, 2018])
    st.divider()
    st.write("LLM:", "✅ Claude configurado" if has_llm() else "⚠️ sem chave — só recuperação")

filters = {}
if score != (1, 5):
    filters.update(min_score=score[0], max_score=score[1])
if cats:
    filters["category"] = cats
if delivery:
    filters["delivery_status"] = delivery
if years:
    filters["purchase_year"] = years

ex = st.selectbox("Exemplos", [""] + EXAMPLES)
question = st.text_input("Pergunta", value=ex)

if st.button("Perguntar", type="primary") and question.strip():
    # filtro explícito de nota tem prioridade sobre a inferência a partir da pergunta
    with st.spinner("Recuperando avaliações e gerando resposta..."):
        res = answer(question, filters, infer_filters=infer and score == (1, 5))

    plan, stats = res["plan"], res["stats"]
    with st.expander("Como a pergunta foi processada", expanded=False):
        st.json({"texto_busca": plan.search_text, "filtros": plan.filters,
                 "regras_aplicadas": plan.notes, "recuperacao": stats})

    if res["status"] == "ok":
        st.markdown(res["answer"])
    elif res["status"] == "retrieval_only":
        st.info(res["answer"])
    else:
        st.warning(res["answer"])

    if res["evidences"]:
        st.subheader("Evidências recuperadas")
        rows = [{"ref": e.ref, "citada": "✔" if e.ref in res["cited"] else "",
                 "similaridade": e.similarity, "nota": e.metadata["review_score"],
                 "categoria": e.metadata["category"], "entrega": e.metadata["delivery_status"],
                 "compra": e.metadata["purchase_month"], "avaliação": e.text,
                 "review_id": e.review_id} for e in res["evidences"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
