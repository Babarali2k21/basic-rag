"""
RAGAS Evaluation Script
────────────────────────
Evaluates the RAG pipeline on a small QA dataset and prints metrics.

Usage:
    python evaluate.py

Metrics produced:
    - faithfulness       (are answers grounded in the retrieved context?)
    - answer_relevancy   (does the answer address the question?)
    - context_precision  (are retrieved chunks actually relevant?)
    - context_recall     (did we retrieve all necessary information?)
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from data_loader import load_documents, split_documents
from vector_store import build_vector_store, get_retriever
from rag import build_rag_chain, ask

# ---------------------------------------------------------------------------
# Evaluation dataset
# Update these Q&A pairs to match YOUR article content
# ---------------------------------------------------------------------------

EVAL_DATA = [
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "a retrieval system with a language model. The retrieval system "
            "fetches relevant documents from a knowledge base, and the language "
            "model uses that context to generate grounded, accurate answers."
        ),
    },
    {
        "question": "What are the main components of a RAG system?",
        "ground_truth": (
            "A RAG system consists of three main components: a document store "
            "or knowledge base, a retriever that finds relevant chunks using "
            "vector similarity search, and a language model that generates "
            "answers conditioned on the retrieved context."
        ),
    },
    {
        "question": "What is a vector database used for in RAG?",
        "ground_truth": (
            "A vector database stores document embeddings and enables fast "
            "similarity search. When a user asks a question, the query is "
            "embedded and compared against stored vectors to retrieve the most "
            "semantically relevant document chunks."
        ),
    },
]


def run_evaluation() -> None:
    print("🔄 Indexing documents for evaluation...")
    docs = load_documents()
    chunks = split_documents(docs)
    vector_store = build_vector_store(chunks)
    retriever = get_retriever(vector_store)
    chain = build_rag_chain(retriever)

    print("🔄 Running RAG pipeline on evaluation questions...")
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in EVAL_DATA:
        result = ask(item["question"], retriever, chain)
        retrieved_docs = retriever.invoke(item["question"])

        questions.append(item["question"])
        answers.append(result.answer)
        contexts.append([doc.page_content for doc in retrieved_docs])
        ground_truths.append(item["ground_truth"])

        print(f"  ✓ Q: {item['question'][:60]}...")

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    print("\n📊 Running RAGAS evaluation...\n")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # Pretty-print results
    print("=" * 50)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 50)
    scores = results.to_pandas()
    for metric in ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]:
        if metric in scores.columns:
            mean_score = scores[metric].mean()
            bar = "█" * int(mean_score * 20)
            print(f"  {metric:<22} {mean_score:.3f}  {bar}")
    print("=" * 50)
    print()

    return results


if __name__ == "__main__":
    run_evaluation()
