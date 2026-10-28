import re
import hashlib
import streamlit as st
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

RETRIEVAL_POOL = 20
GITHUB_URL = "https://github.com/jenniferlinet/rteb"


def _cloud_deploy_mode():
    if os.environ.get("RTEB_CLOUD", "").lower() in ("1", "true", "yes"):
        return True
    try:
        flag = st.secrets.get("RTEB_CLOUD", "0")
        return str(flag).lower() in ("1", "true", "yes")
    except Exception:
        return False


def _embedding_model_name():
    if os.environ.get("EMBEDDING_MODEL"):
        return os.environ["EMBEDDING_MODEL"]
    return (
        "sentence-transformers/all-MiniLM-L6-v2"
        if _cloud_deploy_mode()
        else "all-mpnet-base-v2"
    )


def _reranker_model_name():
    if os.environ.get("RERANKER_MODEL"):
        return os.environ["RERANKER_MODEL"]
    return "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_SUMMARY_SENTENCES = 5
MIN_SENTENCE_CHARS = 28

# -------------------------------------------------------------
# PDF Upload
# -------------------------------------------------------------
st.title("RTEB Retrieval Dashboard")
st.caption(
    "Upload a PDF · compare Keyword, Semantic, and Hybrid search · "
    "extractive answers with cross-encoder reranking"
)
uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])

PDF_PATH = "uploaded_pdf.pdf"

if uploaded_pdf:
    with open(PDF_PATH, "wb") as f:
        f.write(uploaded_pdf.read())
    st.session_state.pop("llm_eval_cache", None)
    st.success("PDF uploaded successfully!")


def normalize_text(text):
    for old, new in (
        ("\ufb01", "fi"),
        ("\ufb02", "fl"),
        ("ﬁ", "fi"),
        ("ﬂ", "fl"),
    ):
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def pdf_fingerprint():
    if not os.path.exists(PDF_PATH):
        return "missing"
    stat = os.stat(PDF_PATH)
    return f"{stat.st_mtime_ns}-{stat.st_size}"


# -------------------------------------------------------------
# Load PDF
# -------------------------------------------------------------
@st.cache_resource
def load_pdf_docs(_pdf_fp):
    if _pdf_fp == "missing":
        return []

    loader = PyPDFLoader(PDF_PATH)
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(raw_docs)
    for doc in chunks:
        doc.page_content = normalize_text(doc.page_content)
    return [c for c in chunks if c.page_content]


pdf_fp = pdf_fingerprint()
pdf_docs = load_pdf_docs(pdf_fp)
DOC_TEXTS = [doc.page_content for doc in pdf_docs]


# -------------------------------------------------------------
# Embedding + reranker
# -------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=_embedding_model_name())


@st.cache_resource
def load_reranker():
    return CrossEncoder(_reranker_model_name())


def get_embedding_model():
    return load_embedding_model()


def get_reranker():
    return load_reranker()


# -------------------------------------------------------------
# FAISS Index
# -------------------------------------------------------------
def faiss_index_path(_pdf_fp):
    digest = hashlib.sha256(_pdf_fp.encode()).hexdigest()[:12]
    return f"./dbs/documentation/faiss_{digest}"


@st.cache_resource
def get_faiss_index(_pdf_fp):
    if _pdf_fp == "missing":
        return None

    index_path = faiss_index_path(_pdf_fp)
    if os.path.exists(index_path):
        try:
            return FAISS.load_local(
                index_path,
                get_embedding_model(),
                allow_dangerous_deserialization=True,
            )
        except Exception:
            pass

    if DOC_TEXTS:
        db = FAISS.from_texts(DOC_TEXTS, get_embedding_model())
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        db.save_local(index_path)
        return db

    return None


# -------------------------------------------------------------
# Relevance evaluation (cross-encoder — stable vs small LLM)
# -------------------------------------------------------------
def score_to_five(raw_score, peer_scores):
    peers = list(peer_scores)
    if len(peers) <= 1:
        norm = 1 / (1 + np.exp(-float(raw_score)))
        return int(np.clip(np.round(norm * 4 + 1), 1, 5))

    lo, hi = min(peers), max(peers)
    if hi == lo:
        norm = 1.0
    else:
        norm = (float(raw_score) - lo) / (hi - lo)
    return int(np.clip(np.round(norm * 4 + 1), 1, 5))


def verdict_from_score(score):
    return {
        5: "Highly Relevant",
        4: "Relevant",
        3: "Partially Relevant",
        2: "Weak",
        1: "Irrelevant",
    }[score]


def query_term_overlap(query, passage):
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "to", "for", "of",
        "in", "on", "at", "and", "or", "do", "i", "my", "how", "what",
        "when", "where", "why", "can", "with", "from", "it", "this",
    }
    q = {
        w for w in re.findall(r"[a-z0-9]+", query.lower())
        if w not in stop and len(w) > 2
    }
    p = set(re.findall(r"[a-z0-9]+", passage.lower()))
    return sorted(q & p)


def build_analysis(query, passage, rerank_raw, keyword_sim, semantic_sim, score):
    overlap = query_term_overlap(query, passage)
    overlap_txt = ", ".join(overlap[:6]) if overlap else "few exact word matches"
    strength = (
        "strong" if score >= 4 else "moderate" if score == 3 else "weak"
    )
    return (
        f"This passage is a **{strength}** match for your question. "
        f"Cross-encoder relevance: {rerank_raw:.2f}. "
        f"Semantic similarity: {semantic_sim:.2f}, keyword similarity: {keyword_sim:.2f}. "
        f"Shared terms include: {overlap_txt}."
    )


def evaluate_passage(query, passage, peer_rerank_scores, keyword_sim, semantic_sim):
    rerank_raw = float(get_reranker().predict([(query, passage)])[0])
    score = score_to_five(rerank_raw, peer_rerank_scores)
    return {
        "llm_score": score,
        "verdict": verdict_from_score(score),
        "reason": build_analysis(
            query, passage, rerank_raw, keyword_sim, semantic_sim, score
        ),
        "rerank_raw": rerank_raw,
    }


def get_cached_evaluation(query, passage, peer_rerank_scores, keyword_sim, semantic_sim):
    if "llm_eval_cache" not in st.session_state:
        st.session_state.llm_eval_cache = {}
    key = (query, passage)
    if key not in st.session_state.llm_eval_cache:
        st.session_state.llm_eval_cache[key] = evaluate_passage(
            query, passage, peer_rerank_scores, keyword_sim, semantic_sim
        )
    return st.session_state.llm_eval_cache[key]


# -------------------------------------------------------------
# Retrieval helpers
# -------------------------------------------------------------
def compute_similarity_vectors(query, doc_texts):
    nq = normalize_text(query)
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(doc_texts)
    q_vec = vectorizer.transform([nq])
    keyword_similarities = cosine_similarity(q_vec, doc_vectors)[0]

    embed = get_embedding_model()
    doc_embeds = embed.embed_documents(doc_texts)
    query_embed = embed.embed_query(nq)
    semantic_similarities = cosine_similarity([query_embed], doc_embeds)[0]

    kw_norm = (keyword_similarities - keyword_similarities.min())
    if keyword_similarities.max() > keyword_similarities.min():
        kw_norm /= keyword_similarities.max() - keyword_similarities.min()
    sem_norm = (semantic_similarities - semantic_similarities.min())
    if semantic_similarities.max() > semantic_similarities.min():
        sem_norm /= semantic_similarities.max() - semantic_similarities.min()

    combined = 0.5 * kw_norm + 0.5 * sem_norm
    return keyword_similarities, semantic_similarities, combined


def candidate_indices(method, keyword_sim, semantic_sim, combined, pool_size):
    if method == "Keyword":
        scores = keyword_sim
    elif method == "Semantic":
        scores = semantic_sim
    else:
        scores = combined
    pool_size = min(pool_size, len(scores))
    return np.argsort(scores)[::-1][:pool_size], scores


def build_retrieval_results(method, query, keyword_sim, semantic_sim, combined, top_k):
    pool_size = min(RETRIEVAL_POOL, len(DOC_TEXTS))
    indices, score_vector = candidate_indices(
        method, keyword_sim, semantic_sim, combined, pool_size
    )
    pairs = [(query, DOC_TEXTS[i]) for i in indices]
    rerank_scores = [float(s) for s in get_reranker().predict(pairs)]
    ranked = sorted(
        zip(indices, rerank_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    peer_scores = [s for _, s in ranked]
    results = []
    for idx, rerank_raw in ranked:
        passage = DOC_TEXTS[idx]
        evaluation = get_cached_evaluation(
            query,
            passage,
            peer_scores,
            float(keyword_sim[idx]),
            float(semantic_sim[idx]),
        )
        results.append({
            "text": passage,
            "similarity": float(score_vector[idx]),
            "rerank_raw": rerank_raw,
            **evaluation,
        })
    return results


def summarize_method(results, top_k):
    scores = [r["llm_score"] for r in results]
    strong_hits = sum(1 for s in scores if s >= 4)
    return {
        "avg_score": round(float(np.mean(scores)), 2),
        "strong_hits": strong_hits,
        "top_k": top_k,
    }


def pick_best_method(comparison_rows):
    return max(
        comparison_rows,
        key=lambda r: (r["avg_score"], r["strong_hits"]),
    )


def render_method_comparison(comparison_rows):
    st.subheader("Comparison: Keyword vs Semantic vs Hybrid")
    winner = pick_best_method(comparison_rows)
    labels = {
        "Keyword": "Matches **exact words** in the PDF (good for error codes, names).",
        "Semantic": "Matches **meaning** even if wording differs.",
        "Hybrid": "Combines both word match and meaning.",
    }

    cols = st.columns(3)
    for col, row in zip(cols, comparison_rows):
        method = row["Method"]
        with col:
            badge = " 🏆 Best for this query" if method == winner["Method"] else ""
            st.markdown(f"#### {method}{badge}")
            st.markdown(labels[method])
            st.markdown(
                f"**Average match:** {row['avg_score']}/5  \n"
                f"**Strong passages:** {row['strong_hits']} of {row['top_k']}"
            )

    st.success(
        f"**Recommendation for this query:** use **{winner['Method']}** "
        f"(highest average match: {winner['avg_score']}/5)."
    )


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []
    for part in parts:
        sent = normalize_text(part)
        if len(sent) >= MIN_SENTENCE_CHARS:
            sentences.append(sent)
    return sentences


def extractive_answer_summary(query, retrieved_docs, max_sentences=MAX_SUMMARY_SENTENCES):
    candidates = []
    seen = set()

    for passage_rank, doc in enumerate(retrieved_docs):
        for sentence_idx, sent in enumerate(split_sentences(doc["text"])):
            dedupe_key = sent.lower()[:100]
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append({
                "passage_rank": passage_rank,
                "sentence_idx": sentence_idx,
                "text": sent,
            })

    if not candidates:
        return "", []

    pairs = [(query, c["text"]) for c in candidates]
    scores = get_reranker().predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate["score"] = float(score)

    max_score = max(c["score"] for c in candidates)
    score_floor = max_score - 1.25
    strong = [c for c in candidates if c["score"] >= score_floor]
    if not strong:
        strong = sorted(candidates, key=lambda c: c["score"], reverse=True)[
            :max_sentences
        ]

    selected = sorted(strong, key=lambda c: c["score"], reverse=True)[:max_sentences]
    selected.sort(key=lambda c: (c["passage_rank"], c["sentence_idx"]))
    summary = " ".join(c["text"] for c in selected)
    return summary, selected


def render_passage_card(doc, index):
    icon = (
        "✅" if doc["llm_score"] >= 4
        else "⚠️" if doc["llm_score"] == 3
        else "❌"
    )
    st.markdown(f"### {icon} Passage {index}")
    st.write(doc["text"])
    st.markdown(f"- **Retrieval score:** {doc['similarity']:.2f}")
    st.markdown(f"- **Relevance score:** **{doc['llm_score']}/5**")
    st.markdown(f"- **Verdict:** {doc['verdict']}")
    st.markdown(f"- **Analysis:** {doc['reason']}")


# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
has_pdf = os.path.exists(PDF_PATH)
if has_pdf and DOC_TEXTS:
    total_chars = sum(len(t) for t in DOC_TEXTS)
    page_nums = [
        (getattr(d, "metadata", {}) or {}).get("page", 0) for d in pdf_docs
    ]
    page_count = max(page_nums) + 1 if page_nums else 0
    st.caption(
        f"Document indexed: **{page_count}** PDF page(s) · "
        f"**{len(DOC_TEXTS)}** searchable chunks · "
        f"**{total_chars:,}** characters of text extracted."
    )
elif has_pdf and not uploaded_pdf:
    st.caption(f"Using saved PDF: `{PDF_PATH}`.")

query = st.text_input("Enter your query:")
retrieval_method = st.radio(
    "Select retrieval method for detailed results:",
    ["Keyword", "Semantic", "Hybrid"],
)
top_k = st.slider("Select number of top documents (k):", 1, 10, 3)

with st.expander("Tips for better results"):
    st.markdown(
        """
- Ask **specific** questions; use words you expect to appear in the document.
- Try **Hybrid** first; use **Keyword** for exact phrases or error messages.
- The **summary** uses only sentences from your PDF (nothing invented).
        """
    )

if not has_pdf:
    st.info("Upload a PDF above to start (use a public sample doc for demos — do not upload confidential files).")
    st.stop()

if not DOC_TEXTS:
    st.warning("PDF loaded, but text extraction failed. Try a text-based PDF (not a scan).")
    st.stop()


# -------------------------------------------------------------
# Retrieval Logic
# -------------------------------------------------------------
if query:
    with st.spinner("Loading models and retrieving passages…"):
        get_embedding_model()
        get_reranker()
        keyword_sim, semantic_sim, combined_sim = compute_similarity_vectors(
            query, DOC_TEXTS
        )

        methods = ["Keyword", "Semantic", "Hybrid"]
        method_results = {}
        comparison_rows = []

        for method in methods:
            results = build_retrieval_results(
                method, query, keyword_sim, semantic_sim, combined_sim, top_k
            )
            method_results[method] = results
            comparison_rows.append(
                {"Method": method, **summarize_method(results, top_k)}
            )

        retrieved_docs = method_results[retrieval_method]
        summary_text, summary_sentences = extractive_answer_summary(
            query, retrieved_docs
        )

    best = retrieved_docs[0] if retrieved_docs else None

    if summary_text:
        st.subheader("📝 Extractive answer summary")
        st.info(summary_text)
        st.caption(
            f"Composed from **{len(summary_sentences)}** sentence(s) in the PDF only "
            f"({retrieval_method} retrieval). Nothing here is invented."
        )
        with st.expander("View summary source sentences"):
            for i, sent in enumerate(summary_sentences, 1):
                st.markdown(
                    f"{i}. (score {sent['score']:.2f}) {sent['text']}"
                )

    if best:
        st.subheader("💡 Best matching passage (verbatim)")
        st.success(best["text"])
        st.caption(
            f"From **{retrieval_method}** retrieval · "
            f"relevance **{best['llm_score']}/5** · "
            f"reranker {best['rerank_raw']:.2f}"
        )

    render_method_comparison(comparison_rows)

    st.subheader(f"🔍 Retrieved Passages — {retrieval_method}")
    for i, doc in enumerate(retrieved_docs, 1):
        render_passage_card(doc, i)

st.divider()
st.caption(f"[View source on GitHub]({GITHUB_URL})")
