"""
Streamlit web UI for basic-rag.
Run:  streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="DocChat · RAG Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from data_loader import load_documents, split_documents
from vector_store import build_vector_store, load_vector_store, get_retriever
from rag import build_rag_chain, ask, RAGResponse


# ── Session state defaults ───────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "retriever": None,
    "chain": None,
    "indexed": False,
    "num_chunks": 0,
    "indexed_files": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-…",
        help="Used only for this session. Never stored.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.subheader("Knowledge Base")

    source_option = st.radio(
        "Document source",
        ["Upload files", "Use articles/ folder"],
        help="Upload your own documents or use the pre-loaded articles folder.",
    )

    uploaded_files = None
    if source_option == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload .txt or .pdf files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="Upload one or more documents to chat with.",
        )
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected:")
            for f in uploaded_files:
                st.markdown(f"• `{f.name}`")
    else:
        articles_dir = Path("./articles")
        if articles_dir.exists():
            files = list(articles_dir.glob("*.txt")) + list(articles_dir.glob("*.pdf"))
            st.caption(f"{len(files)} file(s) in ./articles:")
            for f in files:
                st.markdown(f"• `{f.name}`")
        else:
            st.warning("No `./articles` folder found.")

    st.divider()

    index_ready = (
        (source_option == "Upload files" and uploaded_files) or
        (source_option == "Use articles/ folder")
    )

    if st.button("Index Documents", use_container_width=True, type="primary",
                 disabled=not index_ready):
        if not api_key:
            st.error("Please enter your OpenAI API key first.")
        else:
            with st.spinner("Loading and indexing documents…"):
                try:
                    if source_option == "Upload files" and uploaded_files:
                        # Save uploaded files to a temp directory
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            for uploaded_file in uploaded_files:
                                file_path = Path(tmp_dir) / uploaded_file.name
                                file_path.write_bytes(uploaded_file.read())

                            docs = load_documents(directory=tmp_dir)
                            chunks = split_documents(docs)
                            vs = build_vector_store(chunks)

                        st.session_state.indexed_files = [f.name for f in uploaded_files]
                    else:
                        docs = load_documents()
                        chunks = split_documents(docs)
                        vs = build_vector_store(chunks)
                        articles_dir = Path("./articles")
                        files = list(articles_dir.glob("*.txt")) + list(articles_dir.glob("*.pdf"))
                        st.session_state.indexed_files = [f.name for f in files]

                    st.session_state.retriever = get_retriever(vs)
                    st.session_state.chain = build_rag_chain(st.session_state.retriever)
                    st.session_state.indexed = True
                    st.session_state.num_chunks = len(chunks)
                    st.session_state.messages = []  # reset chat on new index
                    st.success(f"Indexed {len(chunks)} chunks!")

                except Exception as e:
                    st.error(f"Error during indexing: {e}")

    if st.session_state.indexed:
        st.info(
            f"**Index ready**\n\n"
            f"- Chunks stored: {st.session_state.num_chunks}\n"
            f"- Files: {len(st.session_state.indexed_files)}"
        )
        with st.expander("View indexed files"):
            for f in st.session_state.indexed_files:
                st.markdown(f"• `{f}`")

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**Stack:** LangChain · ChromaDB · OpenAI · Streamlit")
    st.caption("[GitHub →](https://github.com/Babarali2k21/basic-rag)")


# ── Main area ────────────────────────────────────────────────────────────────
st.title("DocChat")
st.caption("Upload your documents and ask questions - answers grounded in your knowledge base.")

if not st.session_state.indexed:
    st.info(
        "**Get started:**\n\n"
        "1. Enter your OpenAI API key\n"
        "2. Upload documents or use the articles folder\n"
        "3. Click **Index Documents**\n"
        "4. Start asking questions!"
    )
    st.stop()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")
        if msg.get("chunks_used") is not None:
            st.caption(f"Chunks retrieved: {msg['chunks_used']}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            try:
                result: RAGResponse = ask(
                    prompt,
                    st.session_state.retriever,
                    st.session_state.chain,
                )
                st.markdown(result.answer)
                if result.sources:
                    with st.expander("Sources used"):
                        for src in result.sources:
                            st.markdown(f"- `{src}`")
                st.caption(f"Chunks retrieved: {result.num_chunks_used}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                    "chunks_used": result.num_chunks_used,
                })
            except Exception as e:
                error_msg = f"Something went wrong: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })