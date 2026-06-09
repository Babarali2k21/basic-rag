from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def build_vector_store(chunks: List[Document]) -> Chroma:
    """
    Embed chunks and persist them in ChromaDB.
    Safe to call multiple times — recreates the collection each run.
    """
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB "
          f"→ {settings.chroma_persist_dir}")
    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing persisted ChromaDB collection.
    Raises RuntimeError if the collection doesn't exist yet.
    """
    import os
    if not os.path.exists(settings.chroma_persist_dir):
        raise RuntimeError(
            "ChromaDB not found. Run indexing first: "
            "python main.py --index"
        )
    embeddings = get_embeddings()
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def get_retriever(vector_store: Chroma):
    """Return a LangChain retriever ready for use in the RAG chain."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retrieval_k},
    )
