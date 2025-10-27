import chromadb

from config import Config

class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(Config.CHROMA_PATH)
        embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=Config.OPENAI_API_KEY, model_name=Config.EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            embedding_function=embedding_fn
        )

    def upsert(self, docs):
        self.collection.upsert(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            
        )

    def query(self, query, n_results=3):
        results = self.collection.query(query_text=query, n_results=n_results) 
        return results["documents"]
