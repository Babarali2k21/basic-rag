import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def load_documents(directory: str | None = None) -> List[Document]:
    """
    Load .txt and .pdf files from a directory.
    Returns a list of LangChain Document objects with metadata.
    """
    directory = directory or settings.articles_dir
    docs: List[Document] = []
    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(f"Articles directory not found: {directory}")

    for file in path.iterdir():
        if file.suffix == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
            docs.extend(loader.load())
        elif file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())

    if not docs:
        raise ValueError(f"No .txt or .pdf files found in: {directory}")

    print(f"✅ Loaded {len(docs)} document(s) from {directory}")
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    Preserves source metadata on every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks "
          f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})")
    return chunks