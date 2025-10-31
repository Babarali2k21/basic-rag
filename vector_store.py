import chromadb
from chromadb.utils import embedding_functions
from config import Config

config = Config()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=config.openai_key,
    model_name=config.embedding_model
)

chroma_client = chromadb.PersistentClient(path=config.chroma_path)
collection = chroma_client.get_or_create_collection(
    name=config.collection_name,
    embedding_function=openai_ef
)

def upsert_documents(chunked_documents):
    for doc in chunked_documents:
        collection.upsert(
            ids=[doc["id"]],
            documents=[doc["text"]],
            embeddings=[doc["embedding"]]
        )

def query_documents(question, n_results=2):
    results = collection.query(query_texts=[question], n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    return relevant_chunks