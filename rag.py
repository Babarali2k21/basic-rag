from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import settings

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- If the answer is clearly in the context, answer concisely and accurately.
- If the answer is not in the context, say: "I don't have enough information to answer that based on the provided documents."
- Always mention which document your answer comes from when possible.
- Do not make up information.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])

# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    num_chunks_used: int = 0

# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        label = f"{source}" + (f" (page {page + 1})" if page != "" else "")
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_sources(docs: List[Document]) -> List[str]:
    seen: set = set()
    sources: List[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_rag_chain(retriever):
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
    )
    return chain


def ask(question: str, retriever, chain) -> RAGResponse:
    relevant_docs: List[Document] = retriever.invoke(question)
    ai_message = chain.invoke(question)
    answer: str = ai_message.content

    return RAGResponse(
        question=question,
        answer=answer,
        sources=_extract_sources(relevant_docs),
        num_chunks_used=len(relevant_docs),
    )