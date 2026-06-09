"""
basic-rag CLI
─────────────
Usage:
    python main.py --index          # Index documents in ./articles
    python main.py                  # Interactive Q&A (uses existing index)
    python main.py --query "..."    # Single question, then exit
"""

import argparse

from data_loader import load_documents, split_documents
from vector_store import build_vector_store, load_vector_store, get_retriever
from rag import build_rag_chain, ask


def index_documents() -> None:
    docs = load_documents()
    chunks = split_documents(docs)
    build_vector_store(chunks)
    print("\n✅ Indexing complete. Run `python main.py` to start Q&A.\n")


def run_interactive(query: str | None = None) -> None:
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    chain = build_rag_chain(retriever)

    def _answer(question: str) -> None:
        result = ask(question, retriever, chain)
        print(f"\n💡 Answer:\n{result.answer}")
        if result.sources:
            print(f"\n📄 Sources: {', '.join(result.sources)}")
        print(f"   Chunks used: {result.num_chunks_used}\n")

    if query:
        _answer(query)
        return

    print("\n🤖 RAG Q&A ready. Type your question or 'exit' to quit.\n")
    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        _answer(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="basic-rag CLI")
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index documents from ./articles before querying",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a single question and exit",
    )
    args = parser.parse_args()

    if args.index:
        index_documents()
    else:
        run_interactive(query=args.query)


if __name__ == "__main__":
    main()
