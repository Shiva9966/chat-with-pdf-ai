import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import tempfile

# Load API key
load_dotenv()

# ---------- SESSION STATE ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Settings")

    st.session_state.dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# ---------- UI CONFIG ----------
st.set_page_config(
    page_title="Chat with PDFs",
    page_icon="📄",
    layout="wide"
)

if st.session_state.dark_mode:
    st.markdown("""
    <style>
    body { background-color: #0e1117; color: white; }
    .chat-box { background-color: #1c1f26; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .chat-box {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📄 Chat with Multiple PDFs")
st.caption("Upload PDFs and chat with AI — with sources & history")

# ---------- PDF UPLOAD ----------
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    documents = []

    with st.spinner("Processing PDFs..."):
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()

                for d in docs:
                    page = d.metadata.get("page", 0) + 1
                    d.metadata["source"] = f"{file.name} (Page {page})"
                    documents.append(d)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    qa_chain = RetrievalQAWithSourcesChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=vectorstore.as_retriever()
    )

    # ---------- CHAT INPUT ----------
    query = st.text_input("Ask a question from the PDFs")

    if query:
        result = qa_chain({"question": query})

        st.session_state.chat_history.append({
            "question": query,
            "answer": result["answer"],
            "sources": result["sources"]
        })

# ---------- CHAT HISTORY ----------
if st.session_state.chat_history:
    st.subheader("💬 Chat History")

    for chat in reversed(st.session_state.chat_history):
        st.markdown(
            f"<div class='chat-box'><b>You:</b> {chat['question']}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='chat-box'><b>AI:</b> {chat['answer']}</div>",
            unsafe_allow_html=True
        )

        if chat["sources"]:
            st.caption(f"📌 Sources: {chat['sources']}")
