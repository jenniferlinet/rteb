import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from openai import AzureOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------------------------------------
# Azure OpenAI Configuration
# -------------------------------------------------------------
AZURE_ENDPOINT = "ENTER-YOUR-AZURE-ENDPOINT"
AZURE_API_KEY = "ENTER-YOUR-AZURE-API-KEY"
AZURE_API_VERSION = "ENTER-YOUR-AZURE-API-VERSION"
GPT_DEPLOYMENT = "ENTER-YOUR-AZURE-GPT-DEPLOYMENT"

os.environ["AZURE_OPENAI_API_KEY"] = AZURE_API_KEY

# -------------------------------------------------------------
# PDF Upload
# -------------------------------------------------------------
st.title("📄 RTEB Retrieval Dashboard (PDF Upload + Hybrid Search)")
uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])

PDF_PATH = "uploaded_pdf.pdf"
INDEX_PATH = "./dbs/documentation/faiss_index"

if uploaded_pdf:
    with open(PDF_PATH, "wb") as f:
        f.write(uploaded_pdf.read())
    st.success("PDF uploaded successfully!")


# -------------------------------------------------------------
# Load PDF
# -------------------------------------------------------------
@st.cache_resource
def load_pdf_docs(pdf_exists):
    if not pdf_exists:
        return []

    loader = PyPDFLoader(PDF_PATH)
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    return text_splitter.split_documents(raw_docs)


pdf_docs = load_pdf_docs(os.path.exists(PDF_PATH))
DOC_TEXTS = [doc.page_content for doc in pdf_docs]


# -------------------------------------------------------------
# Embedding Model
# -------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')


embedding_model = load_embedding_model()


# -------------------------------------------------------------
# FAISS Index (Rebuild when new PDF uploaded)
# -------------------------------------------------------------
@st.cache_resource
def get_faiss_index(pdf_changed):
    if not pdf_changed and os.path.exists(INDEX_PATH):
        try:
            return FAISS.load_local(
                INDEX_PATH, embedding_model,
                allow_dangerous_deserialization=True
            )
        except:
            pass

    if pdf_docs:
        db = FAISS.from_texts(DOC_TEXTS, embedding_model)
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        db.save_local(INDEX_PATH)
        return db

    return None


faiss_index = get_faiss_index(uploaded_pdf is not None)


# -------------------------------------------------------------
# Azure GPT-4.1 Client
# -------------------------------------------------------------
@st.cache_resource
def init_gpt_client():
    return AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )


gpt_client = init_gpt_client()


# -------------------------------------------------------------
# LLM Relevance Scoring
# -------------------------------------------------------------
def llm_relevance_score(doc_text, query):
    prompt = (
        f'Question: "{query}"\n\n'
        f'Passage:\n"""{doc_text}"""\n\n'
        "Rate the relevance of this passage to the question "
        "on a scale of 1–5 with a short justification."
    )
    try:
        response = gpt_client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You evaluate relevance concisely."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0,
        )

        reply = response.choices[0].message.content.strip()
        score_line = next(
            (line for line in reply.splitlines() if "score" in line.lower()),
            None,
        )
        score = 3
        if score_line:
            for char in score_line:
                if char.isdigit():
                    score = int(char)
                    break

        return score, reply

    except Exception as e:
        return 3, f"Error: {e}"


# -------------------------------------------------------------
# Retrieval UI
# -------------------------------------------------------------
query = st.text_input("Enter your query:")
retrieval_method = st.radio(
    "Select retrieval method:", ["Keyword", "Semantic", "Hybrid"]
)
top_k = st.slider("Select number of top documents (k):", 1, 10, 3)

if not uploaded_pdf:
    st.info("Please upload a PDF to begin.")
    st.stop()

if not DOC_TEXTS:
    st.warning("PDF loaded, but text extraction failed.")
    st.stop()


# -------------------------------------------------------------
# Retrieval Logic
# -------------------------------------------------------------
retrieved_docs = []

if query:
    keyword_similarities = None
    semantic_similarities = None

    # --- Keyword TF-IDF ---
    if retrieval_method in ["Keyword", "Hybrid"]:
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(DOC_TEXTS)
        q_vec = vectorizer.transform([query])
        keyword_similarities = cosine_similarity(q_vec, doc_vectors)[0]

    # --- Semantic Embedding ---
    if retrieval_method in ["Semantic", "Hybrid"]:
        doc_embeds = embedding_model.embed_documents(DOC_TEXTS)
        query_embed = embedding_model.embed_query(query)
        semantic_similarities = cosine_similarity([query_embed], doc_embeds)[0]

    # --- Combine ---
    if retrieval_method == "Hybrid":
        combined = 0.5 * keyword_similarities + 0.5 * semantic_similarities
        top_indices = np.argsort(combined)[::-1][:top_k]
    elif retrieval_method == "Keyword":
        top_indices = np.argsort(keyword_similarities)[::-1][:top_k]
    else:
        top_indices = np.argsort(semantic_similarities)[::-1][:top_k]

    # --- Build results ---
    for idx in top_indices:
        passage = DOC_TEXTS[idx]
        sim_score = (
            keyword_similarities[idx]
            if retrieval_method == "Keyword"
            else semantic_similarities[idx]
            if retrieval_method == "Semantic"
            else combined[idx]
        )

        llm_score, llm_comment = llm_relevance_score(passage, query)

        retrieved_docs.append({
            "text": passage,
            "similarity": sim_score,
            "llm_score": llm_score,
            "llm_comment": llm_comment,
        })


# -------------------------------------------------------------
# Display Results
# -------------------------------------------------------------
if query:
    st.subheader("🔍 Retrieved Passages")

    for i, doc in enumerate(retrieved_docs, 1):
        icon = (
            "✅" if doc["llm_score"] >= 4
            else "⚠️" if doc["llm_score"] == 3
            else "❌"
        )

        st.markdown(f"### {icon} Passage {i}")
        st.write(doc["text"])

        st.markdown(f"- **Similarity Score:** {doc['similarity']:.2f}")
        st.markdown(f"- **LLM Score:** {doc['llm_score']}/5")
        st.markdown(f"- **Comment:** {doc['llm_comment']}")
