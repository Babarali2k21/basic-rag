from openai import OpenAI
from config import Config

class RAG:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self, vector_db):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.vector_db = vector_db

    def retrieve(self, question, top_k=3):
        return self.vector_db.query(question, n_results=top_k)

    def generate(self, question, context_chunks):
        context = "\n\n".join(context_chunks)
        system_prompt = (
            "You are a helpful assistant. "
            "Answer concisely using the provided context. "
            "If unsure, say 'I don't know'."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

        response = self.client.chat.completions.create(
            model=Config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message["content"]
