import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSmart AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .main-header h1 { color: #e94560; margin: 0; font-size: 2.5rem; }
        .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 1.1rem; }

        .metric-card {
            background: #1e1e2e;
            border: 1px solid #2d2d42;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        .metric-card h3 { color: #e94560; font-size: 1.8rem; margin: 0; }
        .metric-card p  { color: #a8b2d8; margin: 0.3rem 0 0; font-size: 0.85rem; }

        .chat-message-user {
            background: #0f3460;
            border-left: 4px solid #e94560;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            margin: 0.8rem 0;
            color: #e8eaf6;
        }
        .chat-message-ai {
            background: #1e1e2e;
            border-left: 4px solid #00b4d8;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            margin: 0.8rem 0;
            color: #e8eaf6;
        }
        .status-badge {
            display: inline-block;
            background: #00b4d8;
            color: white;
            padding: 0.2rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .stButton > button {
            background: linear-gradient(135deg, #e94560, #c0392b);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #c0392b, #e94560);
            transform: translateY(-1px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>📊 FinSmart AI</h1>
        <p>RAG-based Financial Document Intelligence System</p>
        <p><span class="status-badge">Powered by Llama-3 × LangChain × ChromaDB</span></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    model_choice = st.selectbox(
        "🤖 LLM Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        help="Larger models give better answers but are slower.",
    )
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.0, 0.1,
                            help="0 = deterministic, 1 = creative")
    chunk_size = st.slider("✂️ Chunk Size", 256, 1024, 500, 64,
                           help="Characters per text chunk")
    chunk_overlap = st.slider("🔗 Chunk Overlap", 0, 200, 50, 10)
    k_results = st.slider("🔍 Top-K Retrieval", 1, 8, 4,
                          help="How many chunks are sent to the LLM")

    st.markdown("---")
    st.markdown("### 📂 Upload Financial Report")
    uploaded_file = st.file_uploader(
        "Drop a PDF (10-K, Annual Report, etc.)",
        type=["pdf"],
        help="Upload any company's annual report or financial statement.",
    )

    process_btn = st.button("🚀 Process Document", use_container_width=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Document processing ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def build_qa_chain(file_bytes: bytes, filename: str, cfg: dict) -> tuple:
    """Parse PDF → split → embed → store → build RetrievalQA chain."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = load_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="financial_docs",
    )

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=cfg["model"],
        temperature=cfg["temperature"],
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are FinSmart AI, an expert financial analyst.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I could not find that information in the document."

Context:
{context}

Question: {question}

Answer (be concise and use bullet points where appropriate):""",
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": cfg["k_results"]}),
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
    )

    os.unlink(tmp_path)

    stats = {
        "pages": len(documents),
        "chunks": len(chunks),
        "filename": filename,
    }
    return qa_chain, stats


# ── Trigger processing ────────────────────────────────────────────────────────
if process_btn:
    if uploaded_file is None:
        st.sidebar.error("Please upload a PDF first.")
    elif not groq_api_key:
        st.sidebar.error("GROQ_API_KEY not found in .env")
    else:
        cfg = {
            "model": model_choice,
            "temperature": temperature,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "k_results": k_results,
        }
        with st.spinner("⚙️ Processing document — this may take a moment…"):
            qa_chain, stats = build_qa_chain(
                uploaded_file.read(), uploaded_file.name, cfg
            )
            st.session_state.qa_chain = qa_chain
            st.session_state.doc_stats = stats
            st.session_state.chat_history = []
        st.sidebar.success(f"✅ Ready! {stats['pages']} pages · {stats['chunks']} chunks")

# ── Metrics row ───────────────────────────────────────────────────────────────
if st.session_state.doc_stats:
    s = st.session_state.doc_stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h3>{s["pages"]}</h3><p>Pages Parsed</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h3>{s["chunks"]}</h3><p>Text Chunks</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><h3>{len(st.session_state.chat_history)}</h3>'
            f"<p>Questions Asked</p></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="metric-card"><h3>RAG</h3><p>Active Pipeline</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown(f"**📄 Document:** `{s['filename']}`")
    st.markdown("---")

# ── Chat interface ────────────────────────────────────────────────────────────
if st.session_state.qa_chain:
    st.markdown("### 💬 Ask the Document")

    # Display history
    for msg in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-message-user">🧑 <strong>You:</strong> {msg["question"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="chat-message-ai">🤖 <strong>FinSmart AI:</strong><br>{msg["answer"]}</div>',
            unsafe_allow_html=True,
        )

    # Suggested questions
    st.markdown("**💡 Try asking:**")
    suggestions = [
        "What are the main risk factors?",
        "Summarize the revenue and net income.",
        "What is the company's business overview?",
        "What are the key growth strategies?",
    ]
    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions):
        if col.button(suggestion, key=suggestion):
            st.session_state._prefill = suggestion

    user_question = st.chat_input("Ask a question about the financial report…")

    # Allow suggestion buttons to prefill
    if hasattr(st.session_state, "_prefill"):
        user_question = st.session_state._prefill
        del st.session_state._prefill

    if user_question:
        with st.spinner("🔍 Retrieving and generating answer…"):
            result = st.session_state.qa_chain.invoke({"query": user_question})
            answer = result["result"]
            sources = result.get("source_documents", [])

        st.session_state.chat_history.append(
            {"question": user_question, "answer": answer}
        )

        st.markdown(
            f'<div class="chat-message-user">🧑 <strong>You:</strong> {user_question}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="chat-message-ai">🤖 <strong>FinSmart AI:</strong><br>{answer}</div>',
            unsafe_allow_html=True,
        )

        if sources:
            with st.expander("📚 Source Chunks Used"):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i} — Page {page}:**")
                    st.caption(doc.page_content[:400] + "…")

        st.rerun()

else:
    st.info("👈 Upload a PDF in the sidebar and click **Process Document** to begin.")
    st.markdown(
        """
        ### How it works — RAG Pipeline

        ```
        PDF Upload  ──►  Text Splitting  ──►  Embeddings  ──►  ChromaDB
                                                                     │
        Your Question ──►  Similarity Search  ◄───────────────────────
                                │
                                ▼
                         Top-K Chunks  ──►  Llama-3 (Groq)  ──►  Answer
        ```

        | Step | Technology | Purpose |
        |------|-----------|---------|
        | PDF Parsing | PyPDFLoader | Extract raw text from pages |
        | Text Splitting | RecursiveCharacterTextSplitter | Break into manageable chunks |
        | Embeddings | HuggingFace all-MiniLM-L6-v2 | Convert text → vectors |
        | Vector Store | ChromaDB | Store & retrieve by similarity |
        | LLM | Llama-3 via Groq API | Generate human-readable answer |
        """,
    )
