"""
Unit tests for the basic-rag pipeline.
Run with: pytest tests/ -v
"""

from unittest.mock import MagicMock
from langchain_core.documents import Document

from data_loader import split_documents
from rag import _format_docs, _extract_sources, RAGResponse, ask


# ── data_loader tests ─────────────────────────────────────────────────────────

def test_split_documents_produces_chunks():
    long_text = "word " * 600
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]
    chunks = split_documents(docs)
    assert len(chunks) > 1


def test_split_documents_preserves_metadata():
    docs = [Document(page_content="hello world " * 50,
                     metadata={"source": "sample.txt"})]
    chunks = split_documents(docs)
    for chunk in chunks:
        assert chunk.metadata.get("source") == "sample.txt"


def test_split_short_document_stays_single_chunk():
    docs = [Document(page_content="Short text.", metadata={"source": "a.txt"})]
    chunks = split_documents(docs)
    assert len(chunks) == 1


# ── rag.py helper tests ───────────────────────────────────────────────────────

def test_format_docs_includes_source():
    docs = [
        Document(page_content="Paris is the capital of France.",
                 metadata={"source": "geo.txt"}),
        Document(page_content="Berlin is the capital of Germany.",
                 metadata={"source": "geo.txt"}),
    ]
    formatted = _format_docs(docs)
    assert "geo.txt" in formatted
    assert "Paris" in formatted
    assert "Berlin" in formatted


def test_format_docs_includes_page_number():
    docs = [Document(page_content="Some text.",
                     metadata={"source": "doc.pdf", "page": 2})]
    formatted = _format_docs(docs)
    assert "page 3" in formatted  # page is 0-indexed internally


def test_extract_sources_deduplicates():
    docs = [
        Document(page_content="chunk 1", metadata={"source": "file_a.txt"}),
        Document(page_content="chunk 2", metadata={"source": "file_a.txt"}),
        Document(page_content="chunk 3", metadata={"source": "file_b.txt"}),
    ]
    sources = _extract_sources(docs)
    assert sources == ["file_a.txt", "file_b.txt"]


def test_extract_sources_unknown_fallback():
    docs = [Document(page_content="chunk", metadata={})]
    sources = _extract_sources(docs)
    assert sources == ["unknown"]


def test_rag_response_dataclass():
    response = RAGResponse(
        question="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation.",
        sources=["rag_intro.txt"],
        num_chunks_used=3,
    )
    assert response.num_chunks_used == 3
    assert "rag_intro.txt" in response.sources


def test_rag_response_default_fields():
    response = RAGResponse(question="Q?", answer="A.")
    assert response.sources == []
    assert response.num_chunks_used == 0


# ── Integration smoke test (mocked) ──────────────────────────────────────────

def test_ask_returns_rag_response():
    mock_doc = Document(
        page_content="RAG combines retrieval with generation.",
        metadata={"source": "rag.txt"},
    )
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(
        content="RAG is a technique combining retrieval and generation."
    )

    result = ask("What is RAG?", mock_retriever, mock_chain)

    assert isinstance(result, RAGResponse)
    assert result.question == "What is RAG?"
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0
    assert result.num_chunks_used == 1
    assert "rag.txt" in result.sources


def test_ask_answer_is_string_not_object():
    """Regression: original code returned message object, not .content string."""
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="The answer is 42.")

    result = ask("Question?", mock_retriever, mock_chain)
    assert isinstance(result.answer, str)


def test_ask_empty_retrieval_handled():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(
        content="I don't have enough information."
    )

    result = ask("Obscure question?", mock_retriever, mock_chain)
    assert result.num_chunks_used == 0
    assert result.sources == []