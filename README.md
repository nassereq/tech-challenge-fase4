# Tech Challenge — Fase 4: Voice of Customer Intelligence com RAG

Solução de **Retrieval-Augmented Generation (RAG)** que transforma as avaliações textuais do
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)
em uma base de conhecimento explorável por perguntas em linguagem natural, com **respostas
fundamentadas em evidências rastreáveis** (cada insight cita as avaliações que o sustentam).

> 🚧 **Status:** estrutura base e pipeline implementados. Pendências na seção [Próximos passos](#próximos-passos).

---

## Sumário

1. [Problema de negócio](#1-problema-de-negócio)
2. [Arquitetura](#2-arquitetura)
3. [Estrutura do repositório](#3-estrutura-do-repositório)
4. [Como executar](#4-como-executar)
5. [Construção da base de conhecimento](#5-construção-da-base-de-conhecimento)
6. [Pipeline RAG](#6-pipeline-rag)
7. [Respostas fundamentadas e tratamento de falta de evidências](#7-respostas-fundamentadas-e-tratamento-de-falta-de-evidências)
8. [Metadados e exploração contextual (desafio adicional)](#8-metadados-e-exploração-contextual-desafio-adicional)
9. [Tecnologias e justificativas](#9-tecnologias-e-justificativas)
10. [Exemplos de perguntas e respostas](#10-exemplos-de-perguntas-e-respostas)
11. [Limitações e evoluções futuras](#11-limitações-e-evoluções-futuras)

---

## 1. Problema de negócio

Gestores de e-commerce acompanham notas de avaliação, mas os insights mais ricos — motivos de
insatisfação, elogios, padrões de entrega e qualidade — estão nos **comentários livres**, inviáveis
de ler manualmente em escala. A solução permite perguntar, por exemplo, *"Quais reclamações
relacionadas à entrega aparecem com maior frequência?"* e receber uma resposta sintetizada **com
as avaliações que a comprovam**.

## 2. Arquitetura

```
                ┌──────────────── Offline (indexação) ────────────────┐
 CSVs Olist ──► data_prep.py ──► review_documents.parquet ──► build_index.py
 (Kaggle)       seleção/limpeza/     (1 doc = 1 avaliação      embeddings e5 +
                metadados            + metadados)              ChromaDB (cosseno)
                └─────────────────────────────────────────────────────┘

                ┌──────────────── Online (consulta) ──────────────────┐
 Pergunta ──► transform_query ──► embedding "query:" ──► busca vetorial + filtros
              (filtros inferidos)                         │
                                    corte por similaridade mínima + dedup + MMR
                                                          │
                     ┌── < MIN_EVIDENCES ──► "sem evidências suficientes" (sem LLM)
                     └── ok ──► contexto <evidencias> ──► Claude ──► resposta com [R#]
                                                                    + tabela de evidências
                └─────────────────────────────────────────────────────┘
```

## 3. Estrutura do repositório

```
tech-challenge-fase4/
├── app.py                  # interface Streamlit (perguntas, filtros, evidências)
├── requirements.txt
├── .env.example            # ANTHROPIC_API_KEY e parâmetros opcionais
├── scripts/
│   └── download_data.py    # download do dataset via kagglehub
├── src/
│   ├── config.py           # caminhos, modelos e hiperparâmetros do RAG
│   ├── data_prep.py        # construção da base de conhecimento
│   ├── embeddings.py       # carregamento do modelo de embeddings
│   ├── build_index.py      # geração de embeddings e indexação no ChromaDB
│   ├── retriever.py        # transformação da pergunta, busca, limiar, MMR
│   ├── generator.py        # construção do contexto e geração (Claude)
│   └── rag.py              # orquestração + CLI
├── data/                   # raw/, processed/, chroma/ (não versionados)
├── notebooks/              # análise exploratória e avaliação (a fazer)
└── reports/                # exemplos de perguntas/respostas e avaliação
```

## 4. Como executar

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env                              # e preencha ANTHROPIC_API_KEY

python scripts/download_data.py                     # 1. baixa os CSVs para data/raw
python -m src.data_prep                             # 2. gera data/processed/review_documents.parquet
python -m src.build_index                           # 3. embeddings + índice ChromaDB (CPU: ~10 min)

python -m src.rag "Quais são os principais problemas relatados pelos clientes?"   # CLI
streamlit run app.py                                                              # interface
```

Sem `ANTHROPIC_API_KEY`, o sistema funciona em **modo somente recuperação** (exibe as evidências).

## 5. Construção da base de conhecimento

| Etapa | Decisão | Justificativa |
|---|---|---|
| Seleção | Apenas avaliações com comentário (`review_comment_message`) com ≥ 3 caracteres e ao menos uma letra | A maioria das avaliações não tem texto; sem texto não há "voz do cliente" a recuperar |
| Ausentes | Título/mensagem nulos → `""`; título vazio é omitido do documento; metadados nulos recebem rótulos explícitos (`desconhecida`, `nao_entregue`, `-999`) | Evita "nan" no texto embedado e permite filtrar ausentes |
| Limpeza | Normalização Unicode NFC, remoção de quebras de linha e espaços múltiplos | Mantém acentos, caixa e negações ("não chegou"): o modelo multilíngue usa o texto original; stemming/stopwords destruiriam semântica |
| Deduplicação | `review_id` repetido (ligado a vários pedidos) mantido uma vez; textos idênticos deduplicados na recuperação | Evita evidências repetidas inflando um tema |
| Documento | 1 documento = 1 avaliação (`título. mensagem`), sem chunking | Avaliações são curtas; cada uma é uma unidade de evidência citável |
| Metadados | nota, sentimento, datas, categoria (predominante do pedido), UF, status/atraso da entrega | Filtros na recuperação e contexto para o LLM |

**Resultado:** 99.224 avaliações → **40.299 documentos** com texto útil (40,6%).
Notas: 1★ 8.627 · 2★ 2.110 · 3★ 3.500 · 4★ 5.848 · 5★ 20.214.
Entrega: 34.725 no prazo · 3.733 atrasadas · 1.841 não entregues. Categoria desconhecida: 2,8%.
Indexação completa (e5-small, CPU): ~7,5 min.

## 6. Pipeline RAG

1. **Transformação da pergunta** (`transform_query`): normalização e inferência de filtros por
   regras auditáveis (ex.: "insatisfeitos" → nota ≤ 2; "elogios" → nota ≥ 4).
2. **Embedding** com `intfloat/multilingual-e5-small` e prefixo `query:` (modelo assimétrico
   pergunta ↔ passagem).
3. **Recuperação semântica** no ChromaDB (HNSW, cosseno) — 60 candidatos, com filtros de metadados.
4. **Seleção**: corte por similaridade mínima, deduplicação e **MMR** (relevância × diversidade)
   → 15 evidências.
5. **Contexto**: evidências em blocos `<avaliacao ref="R#" nota=... categoria=... entrega=...>`.
6. **Geração** com Claude (`claude-opus-5-5`), instruído a citar `[R#]` em toda afirmação.

## 7. Respostas fundamentadas e tratamento de falta de evidências

- **Guarda 1 (recuperação):** se menos de `MIN_EVIDENCES` avaliações passam no limiar de
  similaridade, o LLM **nem é chamado** — retorna-se uma mensagem de evidência insuficiente.
- **Guarda 2 (geração):** o prompt obriga o modelo a responder `SEM_EVIDENCIAS` quando o contexto
  não sustenta a resposta ou a pergunta está fora do escopo.
- **Rastreabilidade:** citações `[R#]` são extraídas e validadas contra as evidências enviadas; a
  interface mostra a tabela de evidências marcando as efetivamente citadas.

## 8. Metadados e exploração contextual (desafio adicional)

Filtros disponíveis (inferidos ou manuais na interface): nota, sentimento, categoria do produto,
situação da entrega (atrasado/no prazo/não entregue), ano da compra. Restringir o espaço de busca
antes da similaridade aumenta a precisão para perguntas como *"principais problemas nas avaliações
com nota baixa"*. *(Análise comparativa com/sem filtros a preencher.)*

## 9. Tecnologias e justificativas

| Componente | Escolha | Por quê |
|---|---|---|
| Embeddings | `intfloat/multilingual-e5-small` (sentence-transformers) | Bom em português, roda em CPU, gratuito/local |
| Banco vetorial | ChromaDB (persistente) | Simples, local, suporta filtros por metadados |
| LLM | Claude (`claude-opus-5-5`, Anthropic SDK) | Forte aderência a instruções de citação e recusa |
| Interface | Streamlit | Demonstração rápida com filtros e tabela de evidências |
| Dados | pandas + parquet | Base intermediária reprodutível |

## 10. Exemplos de perguntas e respostas

*(A preencher com saídas reais — ver `reports/`.)*

## 11. Limitações e evoluções futuras

- Limiar de similaridade precisa de calibração empírica (e5 gera similaridades comprimidas).
- Respostas descrevem a amostra recuperada, não estatísticas da base inteira.
- Evoluções: busca híbrida (BM25 + vetorial), re-ranking com cross-encoder, multi-query via LLM,
  avaliação sistemática (precisão de recuperação, fidelidade das respostas).

---

## Próximos passos

- [x] Executar download, preparação e indexação completa; registrar números da base
- [x] Script de avaliação (`scripts/evaluate.py`) — saída em `reports/avaliacao_recuperacao.md`
- [ ] **Detecção de fora do escopo:** a similaridade sozinha NÃO separa bem (dentro: 0,86–0,90;
      "fotossíntese": 0,86; só "capital da Austrália" caiu abaixo de 0,80). Opções: re-ranker
      cross-encoder com limiar, ou classificador de relevância/escopo via LLM antes da geração
- [ ] Configurar `ANTHROPIC_API_KEY` no `.env` e rodar `python scripts/evaluate.py --llm`
- [ ] Notebook de EDA das avaliações
- [ ] Avaliar melhorias (híbrido/re-rank/multi-query) e justificar escolhas
- [ ] Roteiro e gravação do vídeo (≤ 7 min)
- [ ] Publicar repositório no GitHub
