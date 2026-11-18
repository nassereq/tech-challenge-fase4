"""Configurações centrais do projeto (caminhos, modelos e parâmetros do RAG)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --- Caminhos ---------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "data" / "chroma"
DOCS_PARQUET = DATA_PROCESSED / "review_documents.parquet"

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"

# --- Embeddings -------------------------------------------------------------
# multilingual-e5: bom desempenho em português, leve o bastante para CPU.
# Exige prefixos "query: " / "passage: " (convenção do treinamento do modelo).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
COLLECTION_NAME = "olist_reviews"

# --- Recuperação ------------------------------------------------------------
CANDIDATES_K = 60          # candidatos trazidos do índice vetorial
FINAL_K = 15               # evidências enviadas ao LLM
MMR_LAMBDA = 0.7           # 1.0 = só relevância; menor = mais diversidade
# Limiares de similaridade (cosseno). O e5 produz similaridades "comprimidas"
# (~0.7-0.9), por isso os valores são altos. CALIBRAR com scripts/evaluate.py.
MIN_SIMILARITY = 0.80      # abaixo disso a avaliação não é considerada evidência
MIN_EVIDENCES = 3          # mínimo de evidências para gerar resposta

# --- Geração ----------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")
LLM_EFFORT = os.getenv("LLM_EFFORT", "medium")
LLM_MAX_TOKENS = 4000
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
