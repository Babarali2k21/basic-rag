from data_loader import DataLoader
from embeddings import EmbeddingGenerator
from vector_store import VectorDB
from rag import RAG

def main():
    loader = DataLoader()
    embedder = EmbeddingGenerator()
    vectordb = VectorDB()
    rag = RAG(vectordb)

    # Step 1: Load docs
    docs = loader.load_from_folder("articles") 
    chunks = []
    for doc in docs:
        for i, piece in enumerate(loader.spllit_text(doc["text"])): 
            chunks.append({
                "id": f"{doc['id']}_chunk{i+1}",
                "text": segment
            })

    # Step 2: Embed docs
    for chunk in chunks:
        chunk["embedding"] = embedder.create(chunk["text"])
    vectordb.upsert(chunks)

    # Step 3: Ask user
    question = input("Ask a question: ")
    relevant = rag.retrieve(question)
    answer = rag.generate(question, relevant)

    print("Answer:", answer)

if __name__ == "__main__":
    main()
