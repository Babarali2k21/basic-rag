from data_loader import load_documents_from_directory, split_text
from embeddings import get_openai_embedding
from vector_store import upsert_documents, query_documents
from rag import generate_response

def main():
    # Load documents
    documents = load_documents_from_directory("./articles")
    print(f"Loaded {len(documents)} documents")

    # Split into chunks
    chunked_documents = []
    for doc in documents:
        chunks = split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

    # Embeddings
    for doc in chunked_documents:
        doc["embedding"] = get_openai_embedding(doc["text"])

    # Store in DB
    upsert_documents(chunked_documents)

    # Interactive query loop
    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        relevant_chunks = query_documents(question)
        answer = generate_response(question, relevant_chunks)
        print(f"\n💡 Answer: {answer}\n")

if __name__ == "__main__":
    main()