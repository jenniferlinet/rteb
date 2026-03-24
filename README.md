# 🔍 RTEB Retrieval Dashboard

> **Hybrid Information Retrieval with LLM-Based Relevance Evaluation**

A complete retrieval + evaluation pipeline that combines keyword search, semantic embeddings, and GPT-4.1-powered relevance scoring — all in an interactive Streamlit dashboard.

---

## 📖 Overview

Modern information retrieval systems fail when query wording differs from document text, or when meaning is implied rather than explicit. This project tackles that problem by implementing and comparing **three retrieval strategies** on PDF-based datasets, then evaluating results using an **LLM-based relevance scorer**.

### Retrieval Methods

| Method | Description | Strength | Weakness |
|--------|-------------|----------|----------|
| **Keyword Search (TF-IDF)** | Matches query words to document words | Fast, interpretable | Fails on vocabulary mismatch |
| **Semantic Search** | Matches meaning using vector similarity (FAISS + Sentence-BERT) | Captures context & paraphrasing | Requires ML models |
| **Hybrid Search** | Weighted fusion of keyword + semantic scores | Best of both worlds | Slight computation overhead |

---

## 🏗️ System Architecture

```
User Query
     ↓
Vector Retrieval Engine (Keyword / Semantic / Hybrid)
     ↓
Top-K Retrieved Documents
     ↓
RTEB Evaluation (LLM Scoring via Azure OpenAI GPT-4.1)
     ↓
Dashboard with Metrics (Precision@K · nDCG · Avg LLM Score)
```

---

## ✨ Features

- 📄 **PDF Ingestion** — Extracts and chunks text from uploaded PDF documents
- 🔑 **TF-IDF Keyword Search** — Fast lexical matching with cosine similarity
- 🧠 **Semantic Search** — Sentence-BERT (`all-mpnet-base-v2`) + FAISS nearest-neighbor retrieval
- ⚖️ **Hybrid Search** — Configurable weighted fusion of keyword and semantic scores
- 🤖 **LLM Relevance Scoring** — GPT-4.1 rates each retrieved chunk 1–5 with justification
- 📊 **Evaluation Metrics** — Precision@K, nDCG@K, and Average LLM Score visualized in dashboard

---

## 📐 Core Concepts

### TF-IDF Keyword Search
TF-IDF measures term importance relative to the document collection. Similarity is computed using **cosine similarity** between query and document vectors.

### Semantic Search
Uses **Sentence-BERT** (`all-mpnet-base-v2`) to encode text into high-dimensional vectors. **FAISS** enables efficient approximate nearest-neighbor search in vector space, capturing meaning even when vocabulary differs.

### Hybrid Search
Combines both scores with a configurable alpha weight:

```
hybrid_score = α × semantic_score + (1 − α) × keyword_score
```

### LLM-Based Relevance Scoring

| Score | Meaning |
|-------|---------|
| **5** | Highly relevant — directly answers the query |
| **4** | Relevant but may lack detail |
| **3** | Partially relevant / related but imprecise |
| **2** | Weak relevance |
| **1** | Irrelevant |

---

## 📊 Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **Precision@K** | Proportion of relevant documents in top-K results |
| **nDCG@K** | Ranking quality — rewards placing relevant docs higher |
| **Average LLM Score** | Overall quality of retrieved results |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jenniferlinet/rteb.git
cd rteb-retrieval-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_DEPLOYMENT_NAME=your-deployment-name
```

### 4. Run the Dashboard

```bash
streamlit run main.py
```

---

## 📦 Requirements

```
streamlit
numpy
sentence-transformers
langchain-huggingface
langchain-community
langchain-text-splitters
faiss-cpu
pypdf
openai>=1.40.0
scikit-learn
python-dotenv
```

---

## 🧩 Use Cases

- 🔎 **Enterprise Knowledge Retrieval** — Search internal documents and wikis
- 💬 **Document QA Systems** — Answer questions from large PDF corpora
- 🤖 **Chatbot Backends** — Ground LLM responses in retrieved context
- 🎓 **LMS Assistants** — Help students find relevant course material
- 🔬 **IR Research** — Benchmark retrieval strategies on custom datasets

---

## 📁 Project Structure

```
rteb-retrieval-dashboard/
├── app.py                 # Main Streamlit dashboard
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed)
├── .env.example           # Template for environment variables
└── README.md
```

---

## ⚠️ Notes

- Make sure your Azure OpenAI resource has access to **GPT-4.1** and the deployment name matches `AZURE_DEPLOYMENT_NAME` in your `.env`.
- FAISS runs on CPU by default (`faiss-cpu`). For large corpora, consider `faiss-gpu`.
- The first run will download the Sentence-BERT model (~420 MB).

---

## 📄 License

This project is licensed under the MIT License.
