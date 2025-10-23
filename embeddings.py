from openai import OpenAI

class EmbeddingGenerator:
    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY) 
        self.model = model

    def create(self, text):
        resp = self.client.embeddings.create(model=self.model, text=text)
        return resp.data[0].embedding