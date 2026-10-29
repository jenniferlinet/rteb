import re
import hashlib
import tempfile
import streamlit as st
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

RETRIEVAL_POOL = 20
GITHUB_URL = "https://github.com/jenniferlinet/rteb"
# Higher weight = ranking follows that method more (reranker alone made all 3 identical)
METHOD_SCORE_WEIGHT = {
    "Keyword": 0.88,
    "Semantic": 0.88,
    "Hybrid": 0.60,
}
DEFAULT_HF_LLM = "HuggingFaceH4/zephyr-7b-beta"
SPELLING_ALIASES = {
    "haemoglobin": "hemoglobin",
    "haem": "hem",
    "fibre": "fiber",
    "colour": "color",
}


def _secrets_file_path():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".streamlit",
        "secrets.toml",
    )


def _streamlit_secret(key, default=None):
    env_val = os.environ.get(key)
    if env_val not in (None, ""):
        return env_val
    if not os.path.exists(_secrets_file_path()):
        return default
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _cloud_deploy_mode():
    flag = _streamlit_secret("RTEB_CLOUD", "0")
    return str(flag).lower() in ("1", "true", "yes")


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
uploaded_pdf = st.file_uploader(
    "Upload a PDF document (this session only — refresh clears it)",
    type=["pdf"],
)

col_clear, col_ocr = st.columns([1, 1])
with col_clear:
    clear_doc = st.button("Clear document")
with col_ocr:
    use_ocr = st.checkbox(
        "OCR for scanned pages / images (slower)",
        help="Runs Tesseract on pages with little embedded text.",
    )


def normalize_text(text):
    for old, new in (
        ("\ufb01", "fi"),
        ("\ufb02", "fl"),
        ("ﬁ", "fi"),
        ("ﬂ", "fl"),
    ):
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def prepare_search_query(query):
    display = normalize_text(query)
    search = display.lower()
    for uk, us in SPELLING_ALIASES.items():
        search = search.replace(uk, us)
    return display, search


def query_terms(search_query):
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "to", "for", "of",
        "in", "on", "at", "and", "or", "do", "does", "did", "i", "my",
        "how", "what", "when", "where", "why", "can", "with", "from",
        "it", "this", "that", "your", "document",
    }
    return {
        w
        for w in re.findall(r"[a-z0-9]+", search_query.lower())
        if w not in stop and len(w) > 2
    }


def is_noise_sentence(sent):
    low = sent.lower()
    if "sample questions" in low:
        return True
    if sent.strip().endswith("?") and len(sent) < 120:
        if re.match(r"^(what|define|name|explain|how|why)\b", low):
            return True
    return False


def sentence_candidates_from_passage(text):
    candidates = []
    seen = set()
    for sent in split_sentences(text):
        for piece in (sent,):
            if is_noise_sentence(piece):
                continue
            key = piece.lower()[:120]
            if key not in seen:
                seen.add(key)
                candidates.append(piece)
        for inner in re.findall(r"\(([^)]+)\)", sent):
            clause = normalize_text(inner)
            if len(clause) < MIN_SENTENCE_CHARS or is_noise_sentence(clause):
                continue
            key = clause.lower()[:120]
            if key not in seen:
                seen.add(key)
                candidates.append(clause)
    return candidates


def build_direct_answer(user_query, retrieved_docs):
    _, search_q = prepare_search_query(user_query)
    terms = query_terms(search_q)
    candidates = []
    for doc in retrieved_docs[:3]:
        for sent in sentence_candidates_from_passage(doc["text"]):
            term_hits = sum(1 for t in terms if t in sent.lower())
            if terms and term_hits == 0:
                continue
            candidates.append({"text": sent, "term_hits": term_hits})

    if not candidates and retrieved_docs:
        fallback = sentence_candidates_from_passage(retrieved_docs[0]["text"])
        candidates = [{"text": s, "term_hits": 0} for s in fallback[:3]]

    if not candidates:
        return None, []

    pairs = [(search_q, c["text"]) for c in candidates]
    scores = get_reranker().predict(pairs)
    for cand, score in zip(candidates, scores):
        cand["score"] = float(score) + (0.75 * cand["term_hits"])

    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    top = ranked[:2]
    focus_terms = [t for t in terms if t in " ".join(x["text"].lower() for x in top)]
    if focus_terms:
        subject = focus_terms[0]
        answer = (
            f"**{subject.capitalize()}** — "
            + " ".join(item["text"] for item in top)
        )
    else:
        answer = " ".join(item["text"] for item in top)
    return answer, top


def ocr_page_text(page):
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    try:
        pix = page.get_pixmap(dpi=150)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(image)
    except Exception:
        return ""


def extract_pdf_pages_from_bytes(pdf_bytes, use_ocr=False, source_name="document"):
    """PyMuPDF first (reliable on Linux/cloud), PyPDF fallback, optional OCR."""
    pages = []
    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc):
            text = page.get_text("text") or ""
            if use_ocr and len(normalize_text(text)) < 40:
                ocr_text = ocr_page_text(page)
                if len(normalize_text(ocr_text)) > len(normalize_text(text)):
                    text = ocr_text
            pages.append(
                Document(
                    page_content=normalize_text(text),
                    metadata={"page": page_num, "source": source_name},
                )
            )
        doc.close()
    except Exception:
        pages = []

    if not any(doc.page_content for doc in pages):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for doc in pages:
                doc.page_content = normalize_text(doc.page_content)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return [doc for doc in pages if doc.page_content]


def build_pdf_chunks(pdf_bytes, use_ocr, source_name="document"):
    if not pdf_bytes:
        return [], "no file"

    raw_docs = extract_pdf_pages_from_bytes(
        pdf_bytes, use_ocr=use_ocr, source_name=source_name
    )
    if not raw_docs:
        return [], "empty extract"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(raw_docs)
    for doc in chunks:
        doc.page_content = normalize_text(doc.page_content)
    chunks = [c for c in chunks if c.page_content]
    engine = "pymupdf+ocr" if use_ocr else "pymupdf/pypdf"
    return chunks, engine


if clear_doc:
    st.session_state.pop("pdf_bytes", None)
    st.session_state.pop("pdf_label", None)
    st.session_state.pop("pdf_upload_hash", None)

if uploaded_pdf is not None:
    file_bytes = bytes(uploaded_pdf.getbuffer())
    upload_hash = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("pdf_upload_hash") != upload_hash:
        st.session_state["pdf_bytes"] = file_bytes
        st.session_state["pdf_label"] = uploaded_pdf.name
        st.session_state["pdf_upload_hash"] = upload_hash
        st.success("PDF loaded for this session.")

pdf_bytes = st.session_state.get("pdf_bytes")
pdf_label = st.session_state.get("pdf_label", "document")
if pdf_bytes:
    with st.spinner("Extracting text from PDF…"):
        pdf_docs, extract_engine = build_pdf_chunks(
            pdf_bytes, use_ocr, source_name=pdf_label
        )
else:
    pdf_docs, extract_engine = [], "no file"

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


def get_hf_token():
    return _streamlit_secret("HF_TOKEN", None)


def hf_llm_model_id():
    return _streamlit_secret("HF_LLM_MODEL", DEFAULT_HF_LLM)


def generate_llm_answer(query, passages):
    """Optional grounded answer via Hugging Face Inference API (needs HF_TOKEN)."""
    token = get_hf_token()
    if not token:
        return None

    from huggingface_hub import InferenceClient

    context = "\n\n".join(
        f"Source {i + 1}:\n{p['text'][:950]}"
        for i, p in enumerate(passages[:3])
    )
    prompt = (
        "Answer the question using ONLY the sources below. "
        "If the sources do not contain the answer, say exactly: "
        "The document does not contain enough information.\n\n"
        f"{context}\n\nQuestion: {query}\n\n"
        "Answer in 2–4 clear sentences. Mention source numbers when helpful."
    )
    try:
        client = InferenceClient(token=token)
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=hf_llm_model_id(),
            max_tokens=240,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"Could not generate LLM answer: {exc}"


def minmax_norm(values):
    arr = np.array(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi <= lo:
        return [1.0] * len(arr)
    return ((arr - lo) / (hi - lo)).tolist()


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
    _, search_q = prepare_search_query(query)
    q = query_terms(search_q)
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
    _, search_q = prepare_search_query(query)
    rerank_raw = float(get_reranker().predict([(search_q, passage)])[0])
    score = score_to_five(rerank_raw, peer_rerank_scores)
    return {
        "llm_score": score,
        "verdict": verdict_from_score(score),
        "reason": build_analysis(
            query, passage, rerank_raw, keyword_sim, semantic_sim, score
        ),
        "rerank_raw": rerank_raw,
    }


# -------------------------------------------------------------
# Retrieval helpers
# -------------------------------------------------------------
def compute_similarity_vectors(query, doc_texts):
    _, nq = prepare_search_query(query)
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
    _, search_q = prepare_search_query(query)
    pool_size = min(RETRIEVAL_POOL, len(DOC_TEXTS))
    indices, score_vector = candidate_indices(
        method, keyword_sim, semantic_sim, combined, pool_size
    )
    index_list = [int(i) for i in indices]
    pairs = [(search_q, DOC_TEXTS[i]) for i in index_list]
    rerank_scores = [float(s) for s in get_reranker().predict(pairs)]
    rerank_norm = dict(zip(index_list, minmax_norm(rerank_scores)))
    method_norm = dict(
        zip(index_list, minmax_norm([score_vector[i] for i in index_list]))
    )
    blend = METHOD_SCORE_WEIGHT[method]

    def rank_score(idx):
        return blend * method_norm[idx] + (1 - blend) * rerank_norm[idx]

    ranked_indices = sorted(index_list, key=rank_score, reverse=True)[:top_k]
    rerank_map = dict(zip(index_list, rerank_scores))
    peer_scores = [rerank_map[idx] for idx in ranked_indices]

    results = []
    for idx in ranked_indices:
        rerank_raw = rerank_map[idx]
        passage = DOC_TEXTS[idx]
        evaluation = evaluate_passage(
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


def render_method_comparison(comparison_rows, method_results):
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
        top = method_results[method][0]["text"] if method_results[method] else ""
        preview = (top[:120] + "…") if len(top) > 120 else top
        with col:
            badge = " 🏆 Best for this query" if method == winner["Method"] else ""
            st.markdown(f"#### {method}{badge}")
            st.markdown(labels[method])
            st.markdown(
                f"**Average match:** {row['avg_score']}/5  \n"
                f"**Strong passages:** {row['strong_hits']} of {row['top_k']}"
            )
            if preview:
                st.caption(f"Top hit: _{preview}_")

    tops = [method_results[m][0]["text"] if method_results[m] else "" for m in method_results]
    if len(set(tops)) == 1 and tops[0]:
        st.warning(
            "All three methods returned the same top passage for this query. "
            "Try a more specific question, exact error text (Keyword), or "
            "rephrase (Semantic)."
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
    _, search_q = prepare_search_query(query)
    candidates = []
    seen = set()

    for passage_rank, doc in enumerate(retrieved_docs):
        for sentence_idx, sent in enumerate(
            sentence_candidates_from_passage(doc["text"])
        ):
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

    pairs = [(search_q, c["text"]) for c in candidates]
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
has_pdf = bool(pdf_bytes)
if has_pdf and DOC_TEXTS:
    total_chars = sum(len(t) for t in DOC_TEXTS)
    page_nums = [
        (getattr(d, "metadata", {}) or {}).get("page", 0) for d in pdf_docs
    ]
    page_count = max(page_nums) + 1 if page_nums else 0
    st.caption(
        f"Document indexed: **{page_count}** PDF page(s) · "
        f"**{len(DOC_TEXTS)}** searchable chunks · "
        f"**{total_chars:,}** characters extracted · engine: **{extract_engine}**"
    )
elif has_pdf:
    st.caption(f"Session document: **{pdf_label}** (cleared on browser refresh).")

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
- **AI answer** needs a free [Hugging Face token](https://huggingface.co/settings/tokens) in Streamlit Secrets as `HF_TOKEN`.
        """
    )

if not has_pdf:
    st.info(
        "Upload a PDF to start. Nothing is stored on the server after refresh."
    )
    st.stop()

if not DOC_TEXTS:
    st.warning("PDF loaded, but text extraction failed.")
    st.markdown(
        """
**Try this:**
1. Enable **OCR for scanned pages** and upload again (needs Tesseract on the server).
2. Use a **text-based PDF** (select/copy text in Preview or Acrobat).
        """
    )
    if use_ocr:
        st.caption(
            "OCR requires `tesseract-ocr` on the host (`packages.txt` on Streamlit)."
        )
    st.stop()


# -------------------------------------------------------------
# Retrieval Logic
# -------------------------------------------------------------
if query:
    try:
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
            direct_answer, direct_sents = build_direct_answer(
                query, retrieved_docs
            )
            summary_text, summary_sentences = extractive_answer_summary(
                query, retrieved_docs
            )
            llm_answer = generate_llm_answer(query, retrieved_docs)
    except Exception as exc:
        st.error(f"Something went wrong while retrieving: {exc}")
        st.stop()

    best = retrieved_docs[0] if retrieved_docs else None

    st.subheader("✅ Answer")
    if direct_answer:
        st.success(direct_answer)
        st.caption(
            f"From your uploaded PDF ({retrieval_method} retrieval). "
            "Text only — not generated by an external LLM."
        )
    else:
        st.warning(
            "Could not build a short answer from this PDF for that question. "
            "Try rephrasing or check the source passage below."
        )

    if llm_answer and not llm_answer.startswith("Could not"):
        with st.expander("Optional AI rephrase (Hugging Face)"):
            st.markdown(llm_answer)

    if best:
        with st.expander("Source passage (verbatim from PDF)"):
            st.write(best["text"])
    if summary_text and summary_text != direct_answer:
        with st.expander("Additional supporting sentences"):
            st.write(summary_text)

    render_method_comparison(comparison_rows, method_results)

    st.subheader(f"🔍 Retrieved Passages — {retrieval_method}")
    for i, doc in enumerate(retrieved_docs, 1):
        render_passage_card(doc, i)

st.divider()
st.caption(f"[View source on GitHub]({GITHUB_URL})")
