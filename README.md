# RTEB Retrieval Dashboard

**Hybrid PDF search** — compare **Keyword (TF-IDF)**, **Semantic (embeddings)**, and **Hybrid** retrieval on any uploaded PDF, with **cross-encoder reranking**, **extractive answer summaries**, and a side-by-side method comparison.

| | |
|---|---|
| **Repository** | [github.com/jenniferlinet/rteb](https://github.com/jenniferlinet/rteb) |
| **Live demo** | Deploy via [Streamlit Community Cloud](DEPLOY.md) — add your `*.streamlit.app` URL here after deploy |
| **Stack** | Streamlit · scikit-learn · LangChain · Hugging Face (embeddings + cross-encoder) |

---

## Latest changes

| Area | What changed |
|------|----------------|
| **No paid APIs** | Removed Azure OpenAI. Relevance uses a free **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) instead of GPT/Claude. |
| **Better retrieval** | Two-stage pipeline: retrieve top **20** candidates → **rerank** → show top **k**. Normalized hybrid of TF-IDF + semantic cosine similarity. |
| **Readable answers** | **Extractive summary** (best PDF sentences only) + **best matching passage** (verbatim chunk). |
| **Method comparison** | Simple **3-column** view: average match (1–5), strong passages count, recommended winner — no charts. |
| **Any PDF** | Upload-driven; text normalization (e.g. `fi` ligatures); chunk size **700** / overlap **150**. |
| **Cloud deploy** | Streamlit Cloud–ready: `RTEB_CLOUD=1` uses lighter embeddings (`all-MiniLM-L6-v2`), CPU PyTorch in `requirements.txt`, lazy model load on first query. |
| **Security** | `.gitignore` excludes uploaded PDFs, FAISS indexes, and secrets — safe for public GitHub. |

---

## How it works

```
PDF upload → text extract & chunk
       ↓
Query → Keyword / Semantic / Hybrid (top-20 candidates each)
       ↓
Cross-encoder rerank → top-k passages
       ↓
Extractive summary + per-passage scores + 3-way comparison
```

### Retrieval methods

| Method | Idea | Best for |
|--------|------|----------|
| **Keyword** | TF-IDF + cosine similarity | Exact phrases, error messages, names |
| **Semantic** | Sentence embeddings (`all-mpnet-base-v2` locally, MiniLM on cloud) | Paraphrases, different wording |
| **Hybrid** | 50/50 normalized keyword + semantic | General questions (default choice) |

### Relevance score (1–5)

Cross-encoder scores each query–passage pair. Scores are mapped to 1–5 within the result set, with a short **analysis** (match strength, similarities, shared terms).

| Score | Meaning |
|-------|---------|
| **5** | Directly answers the question |
| **4** | Clearly relevant |
| **3** | Related but incomplete |
| **2** | Weak link |
| **1** | Not useful for the question |

---

## Features

- PDF upload and indexing (page/chunk/character stats)
- Three retrieval strategies on the **same query**
- Cross-encoder reranking for stable ranking
- Extractive answer summary (no hallucinated facts)
- Plain-language comparison: Keyword vs Semantic vs Hybrid
- Optional hosting on **Streamlit Community Cloud** ([DEPLOY.md](DEPLOY.md))

---

## Local setup

### 1. Clone

```bash
git clone https://github.com/jenniferlinet/rteb.git
cd rteb
```

### 2. Install

```bash
python3 -m pip install -r requirements.txt
```

First run downloads embedding and reranker models (~500 MB–1 GB depending on models).

### 3. Run

```bash
streamlit run main.py
```

Open the URL shown (usually `http://localhost:8501`), upload a **text-based PDF**, and ask a specific question.

### Optional environment variables

Copy `.env.example` to `.env` only if you need overrides:

| Variable | Purpose |
|----------|---------|
| `RTEB_CLOUD` | Set to `1` for lighter embeddings (Streamlit free tier) |
| `EMBEDDING_MODEL` | Hugging Face embedding model id |
| `RERANKER_MODEL` | Hugging Face cross-encoder model id |

---

## Deploy (resume / LinkedIn)

See **[DEPLOY.md](DEPLOY.md)** for Streamlit Community Cloud steps.

Quick checklist:

1. Push `main` to GitHub (no confidential PDFs in the repo).
2. [share.streamlit.io](https://share.streamlit.io) → Create app → `main.py`.
3. Secrets: `RTEB_CLOUD = "1"`.
4. Add the public URL to this README, your resume, and LinkedIn Featured.

**Demo tip:** Use a public sample PDF only; visitors can upload their own file in the app.

---

## Project structure

```
rteb/
├── main.py                 # Streamlit app
├── requirements.txt        # Python dependencies (CPU PyTorch)
├── DEPLOY.md               # Streamlit Cloud guide
├── .streamlit/
│   ├── config.toml         # Theme & server settings
│   └── secrets.toml.example
├── .env.example            # Optional model overrides
├── .gitignore              # PDFs, indexes, secrets
└── README.md
```

---

## Requirements

See `requirements.txt`. Main libraries: `streamlit`, `sentence-transformers`, `langchain-huggingface`, `langchain-community`, `faiss-cpu`, `pypdf`, `scikit-learn`, `torch` (CPU).

---

## Use cases

- Compare IR strategies on course readings or papers  
- Portfolio / interview demo for information retrieval  
- Prototype document search before building a production RAG stack  

---

## Notes

- **Text PDFs only** — scanned images need OCR first.  
- **Large files** — very long PDFs are slower on free cloud tiers.  
- **Answers are extractive** — the app quotes the document; it does not generate new facts.  

---

## License

MIT License
